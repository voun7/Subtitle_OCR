import cv2 as cv
import numpy as np
import pyclipper
from shapely.geometry import Polygon

np.seterr(divide='ignore', invalid='ignore')


class MakeBorderAndShrinkMap:
    def __init__(self, shrink_ratio: float = 0.4, thresh_min: float = 0.3, thresh_max: float = 0.7) -> None:
        self.shrink_ratio = shrink_ratio
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max

    def draw_thresh_map(self, polygon: list, canvas: np.ndarray, mask: np.ndarray) -> None:
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

    def __call__(self, data: dict) -> None:
        image_height, image_width = data["image"].shape[1:]
        anns = [ann for ann in data["bboxes"] if Polygon(ann).buffer(0).is_valid]
        gt = np.zeros((image_height, image_width), dtype=np.float32)
        mask = np.ones((image_height, image_width), dtype=np.float32)
        thresh_map = np.zeros((image_height, image_width), dtype=np.float32)
        thresh_mask = np.zeros((image_height, image_width), dtype=np.float32)

        for ann in anns:
            poly = np.array(ann)
            polygon = Polygon(poly)

            # generate gt and mask
            if polygon.area < 1:
                cv.fillPoly(mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
                continue
            else:
                distance = polygon.area * (1 - np.power(self.shrink_ratio, 2)) / polygon.length
                subject = [tuple(_l) for _l in ann]
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
            self.draw_thresh_map(ann, thresh_map, thresh_mask)

        thresh_map = thresh_map * (self.thresh_max - self.thresh_min) + self.thresh_min
        data.update({"shape": [image_height, image_width], "shrink_map": gt, "shrink_mask": mask,
                     "threshold_map": thresh_map, "threshold_mask": thresh_mask})
