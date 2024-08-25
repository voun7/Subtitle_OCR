import cv2 as cv
import imgaug.augmenters as iaa
import numpy as np
from imgaug.augmentables import Keypoint, KeypointsOnImage
from torch.utils.data import Dataset

from sub_ocr.utils import read_image, rescale, normalize_img, crop_image
from .collate_fn import Collator
from .data_source import load_data
from .preprocess import rec_label_ops as enc
from .preprocess.border_shrink_map import MakeBorderAndShrinkMap

_ = [MakeBorderAndShrinkMap]


def create_preprocesses(config: dict, lang: str = None) -> list:
    processes, preprocesses = [], config.get("preprocesses", [])
    for process in preprocesses:
        for name, param in process.items():
            if lang and "LabelEncode" in name:
                param["lang"] = lang
                ops = getattr(enc, name)(**param)
                processes.append(ops)
            else:
                ops = eval(name)() if param is None else eval(name)(**param)
                processes.append(ops)
    return processes


def resize_norm_img(image: np.ndarray, height: int, width: int) -> tuple:
    """
    Image scaling and normalization for dataset. The aspect ratio of the image does not change.
    Padding will be added to reach target height and width.
    :param image: image to be resized.
    :param height: target height for resized image.
    :param width: target width for resized image.
    :return: resized normalized image ([H, W, C] to [C, H, W]) and the rescale value
    """
    scale = min(height / image.shape[0], width / image.shape[1])  # Calculate the scaling factor to resize the image
    resized_image = rescale(scale, image)  # Resize the image while maintaining aspect ratio
    pad_h, pad_w = height - resized_image.shape[0], width - resized_image.shape[1]
    resized_image = cv.copyMakeBorder(resized_image, 0, pad_h, 0, pad_w, cv.BORDER_CONSTANT, value=(0, 0, 0))
    return normalize_img(resized_image), scale


class TextDetectionDataset(Dataset):
    def __init__(self, lang: str, data_type: str, config: dict) -> None:
        self.img_data = load_data(lang, "det", data_type)
        self.data_type, self.transform = data_type, self.augmentations()
        self.preprocesses = create_preprocesses(config)
        self.image_height, self.image_width = config["image_height"], config["image_width"]
        if tensor_keys := config.get("tensor_keys"):
            self.collate_fn = Collator(tensor_keys).collate_fn

    @staticmethod
    def augmentations() -> iaa.meta.Sequential:
        augment_seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Affine(scale=(0.7, 0.95), rotate=(-10, 10)),
            iaa.Resize((0.5, 3.0)),
            iaa.AdditiveGaussianNoise(scale=0.1 * 255),
            iaa.Sometimes(0.5, iaa.Crop(percent=(0, 0.1)), iaa.SaltAndPepper(0.1, per_channel=True))
        ], random_order=True)
        return augment_seq

    def data_augmentation(self, image: np.ndarray, image_labels: list) -> tuple:
        aug = self.transform.to_deterministic()
        image_shape = image.shape
        image = aug.augment_image(image)
        augmented_image_labels = []
        for label in image_labels:
            key_points = [Keypoint(x, y) for x, y in label['bbox']]
            key_points = aug.augment_keypoints([KeypointsOnImage(key_points, shape=image_shape)])[0].keypoints
            # min and max are used here to fix out of bound or negative key points
            poly = [(min(max(0, p.x), image.shape[1] - 1), min(max(0, p.y), image.shape[0] - 1)) for p in key_points]
            augmented_image_labels.append({'bbox': poly})
        return image, augmented_image_labels

    def __len__(self) -> int:
        return len(self.img_data)

    def __getitem__(self, idx: int) -> dict:
        image_path, image_labels = self.img_data[idx]
        image, image_height, image_width = read_image(str(image_path))
        if self.data_type == "train":
            image, image_labels = self.data_augmentation(image, image_labels)
        else:
            # the bbox coordinates will not be allowed to be out of bounds or negative to prevent errors.
            image_labels = [{'bbox': [(max(min(x, image_width), 1), max(min(y, image_height), 1)) for x, y in
                                      label["bbox"]]} for label in image_labels]
        image, scale = resize_norm_img(image, self.image_height, self.image_width)
        image_labels = [{"bbox": rescale(scale, bbox=label["bbox"])} for label in image_labels]
        data = {"image_path": str(image_path), "image": image, "bboxes": [ann["bbox"] for ann in image_labels]}
        if self.preprocesses:
            for process in self.preprocesses:
                process(data)
        return data


class TextRecognitionDataset(Dataset):
    def __init__(self, lang: str, data_type: str, config: dict) -> None:
        self.img_data = load_data(lang, "rec", data_type)
        self.data_type, self.transform = data_type, self.augmentations()
        self.preprocesses = create_preprocesses(config, lang)
        self.image_height, self.image_width = config["image_height"], config["image_width"]
        if tensor_keys := config.get("tensor_keys"):
            self.collate_fn = Collator(tensor_keys).collate_fn

    @staticmethod
    def augmentations() -> iaa.meta.Sequential:
        augment_seq = iaa.Sequential([
            iaa.Affine(scale=(0.8, 1.0)),
            iaa.GaussianBlur((0.0, 1.0)),
            iaa.Sometimes(0.5, iaa.Sharpen(alpha=(0.0, 1.0)), iaa.SaltAndPepper(0.1, per_channel=True))
        ], random_order=True)
        return augment_seq

    def __len__(self) -> int:
        return len(self.img_data)

    def __getitem__(self, idx: int) -> dict:
        image_path, image_labels = self.img_data[idx]
        image, image_height, image_width = read_image(str(image_path))
        blank, text = None, image_labels[0]["text"]
        if bbox := image_labels[0]["bbox"]:
            blank, image = crop_image(image, image_height, image_width, bbox)
        if self.data_type == "train":
            image = self.transform.augment_image(image)
        image, text = resize_norm_img(image, self.image_height, self.image_width)[0], " " if blank else text
        data = {"image_path": str(image_path), "image": image, "text": text}
        if self.preprocesses:
            for process in self.preprocesses:
                data = process(data)
                if data is None:
                    return self.__getitem__(np.random.randint(self.__len__()))
        return data
