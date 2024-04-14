import cv2 as cv
import imgaug.augmenters as iaa
import numpy as np
import pyclipper
import torch
from imgaug.augmentables import Keypoint, KeypointsOnImage
from shapely.geometry import Polygon

from utilities.utils import Types

np.seterr(divide='ignore', invalid='ignore')


def default_augmentation() -> iaa.meta.Sequential:
    augment_seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Affine(scale=(0.7, 0.95), rotate=(-10, 10)),
        iaa.Resize((0.5, 3.0)),
        iaa.Crop(percent=(0, 0.1)),
    ])
    return augment_seq


def data_augmentation(image: np.ndarray, anns: list) -> tuple:
    aug = default_augmentation().to_deterministic()
    image_shape = image.shape
    image = aug.augment_image(image)
    new_anns = []
    for ann in anns:
        key_points = [Keypoint(p[0], p[1]) for p in ann['bbox']]
        key_points = aug.augment_keypoints([KeypointsOnImage(key_points, shape=image_shape)])[0].keypoints
        poly = [(min(max(0, p.x), image.shape[1] - 1), min(max(0, p.y), image.shape[0] - 1)) for p in key_points]
        new_ann = {'bbox': poly, 'text': ann['text']}
        new_anns.append(new_ann)
    return image, new_anns


def random_crop():
    pass


def padded_resize(size: int, image: np.ndarray, anns: list) -> tuple:
    h, w, c = image.shape
    scale_w = size / w
    scale_h = size / h
    scale = min(scale_w, scale_h)
    h = int(h * scale)
    w = int(w * scale)
    pad_image = np.zeros((size, size, c), image.dtype)
    pad_image[:h, :w] = cv.resize(image, (w, h))
    new_anns = []
    for ann in anns:
        poly = np.array(ann['bbox']).astype(np.float64)
        poly *= scale
        new_ann = {'bbox': poly.tolist(), 'text': ann['text']}
        new_anns.append(new_ann)
    return pad_image, new_anns


class MakeBorderAndShrinkMap:
    def __init__(self, image_size, shrink_ratio=0.4, thresh_min=0.3, thresh_max=0.7):
        self.image_size = image_size
        self.shrink_ratio = shrink_ratio
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max

    def draw_thresh_map(self, polygon, canvas, mask):
        polygon = np.array(polygon)
        assert polygon.ndim == 2
        assert polygon.shape[1] == 2

        polygon_shape = Polygon(polygon)
        if polygon_shape.area <= 0:
            return
        # Polygon indentation
        distance = polygon_shape.area * (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
        subject = [tuple(pt) for pt in polygon]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        # Calculate the mask
        padded_polygon = np.array(padding.Execute(distance)[0])
        cv.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)

        x_min = padded_polygon[:, 0].min()
        x_max = padded_polygon[:, 0].max()
        y_min = padded_polygon[:, 1].min()
        y_max = padded_polygon[:, 1].max()
        width = x_max - x_min + 1
        height = y_max - y_min + 1

        polygon[:, 0] = polygon[:, 0] - x_min
        polygon[:, 1] = polygon[:, 1] - y_min

        xs = np.broadcast_to(np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
        ys = np.broadcast_to(np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

        distance_map = np.zeros((polygon.shape[0], height, width), dtype=np.float32)
        for i in range(polygon.shape[0]):
            j = (i + 1) % polygon.shape[0]
            # Calculate the distance from point to line
            absolute_distance = self.compute_distance(xs, ys, polygon[i], polygon[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
        distance_map = np.min(distance_map, axis=0)

        x_min_valid = min(max(0, x_min), canvas.shape[1] - 1)
        x_max_valid = min(max(0, x_max), canvas.shape[1] - 1)
        y_min_valid = min(max(0, y_min), canvas.shape[0] - 1)
        y_max_valid = min(max(0, y_max), canvas.shape[0] - 1)

        canvas[y_min_valid:y_max_valid + 1, x_min_valid:x_max_valid + 1] = np.fmax(
            1 - distance_map[
                y_min_valid - y_min:y_max_valid - y_max + height,
                x_min_valid - x_min:x_max_valid - x_max + width],
            canvas[y_min_valid:y_max_valid + 1, x_min_valid:x_max_valid + 1])

    @staticmethod
    def compute_distance(xs, ys, point_1, point_2):
        """
        compute the distance from point to a line
        ys: coordinates in the first axis
        xs: coordinates in the second axis
        point_1, point_2: (x, y), the end of the line
        """
        square_distance_1 = np.square(xs - point_1[0]) + np.square(ys - point_1[1])
        square_distance_2 = np.square(xs - point_2[0]) + np.square(ys - point_2[1])
        square_distance = np.square(point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

        co_sin = (square_distance - square_distance_1 - square_distance_2) / (
                2 * np.sqrt(square_distance_1 * square_distance_2))
        square_sin = 1 - np.square(co_sin)
        square_sin = np.nan_to_num(square_sin)
        result = np.sqrt(square_distance_1 * square_distance_2 * square_sin / square_distance)
        result[co_sin < 0] = np.sqrt(np.fmin(square_distance_1, square_distance_2))[co_sin < 0]
        return result

    def get_maps(self, anns):
        anns = [ann for ann in anns if Polygon(ann['bbox']).buffer(0).is_valid]
        gt = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        mask = np.ones((self.image_size, self.image_size), dtype=np.float32)
        thresh_map = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        thresh_mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)

        for ann in anns:
            poly = np.array(ann['bbox'])
            polygon = Polygon(poly)

            # generate gt and mask
            if polygon.area < 1:
                cv.fillPoly(mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
                continue
            else:
                distance = polygon.area * (1 - np.power(self.shrink_ratio, 2)) / polygon.length
                subject = [tuple(_l) for _l in ann['bbox']]
                padding = pyclipper.PyclipperOffset()
                padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                shrunk = padding.Execute(-distance)

                if len(shrunk) == 0:
                    cv.fillPoly(mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
                    continue
                else:
                    shrunk = np.array(shrunk[0]).reshape(-1, 2)
                    if shrunk.shape[0] > 2 and Polygon(shrunk).buffer(0).is_valid:
                        cv.fillPoly(gt, [shrunk.astype(np.int32)], 1)
                    else:
                        cv.fillPoly(mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
                        continue

            # generate thresh map and thresh mask
            self.draw_thresh_map(ann['bbox'], thresh_map, thresh_mask)

        thresh_map = thresh_map * (self.thresh_max - self.thresh_min) + self.thresh_min
        return gt, mask, thresh_map, thresh_mask


def normalize_image(image: np.ndarray, mean: tuple = (103.939, 116.779, 123.68)) -> np.ndarray:
    image = image.astype(np.float32)
    image[..., 0] -= mean[0]
    image[..., 1] -= mean[1]
    image[..., 2] -= mean[2]
    return image


def db_collate_fn(batch: list) -> dict:
    to_tensor_keys = ("image", "shrink_map", "shrink_mask", "threshold_map", "threshold_mask")
    tensor_batch = {}
    for sample in batch:
        for key, value in sample.items():
            if key in to_tensor_keys:
                value = torch.from_numpy(value)
            tensor_batch.setdefault(key, []).append(value)
    for key, value in tensor_batch.items():
        if key in to_tensor_keys:
            tensor_batch[key] = torch.stack(value)
    return tensor_batch


def db_preprocess(image_path: str, anns: list, data_type: Types.DataType, image_size: int = 640) -> dict:
    image = cv.imread(image_path)
    image_height, image_width, _ = image.shape
    bboxes = [ann["bbox"] for ann in anns]
    if data_type == Types.train:
        image, anns = data_augmentation(image, anns)
        # image, anns = random_crop(image, anns)
    image, anns = padded_resize(image_size, image, anns)
    gen_maps = MakeBorderAndShrinkMap(image_size)
    gt, mask, thresh_map, thresh_mask = gen_maps.get_maps(anns)
    image = normalize_image(image)
    image = np.moveaxis(image, -1, 0)  # change image data format from [H, W, C] to [C, H, W]
    data = {
        "image_path": image_path,
        "image": image,
        "shape": [image_height, image_width],
        "bbox": bboxes,
        "shrink_map": gt,
        "shrink_mask": mask,
        "threshold_map": thresh_map,
        "threshold_mask": thresh_mask,
    }
    return data
