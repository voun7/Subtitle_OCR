from os import environ

environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import albumentations as A
import cv2 as cv
import numpy as np
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
    def augmentations() -> A.Compose:
        augment_seq = A.Compose([
            A.Affine(scale=(0.75, 1.2), rotate=(-10, 10), p=1),
            A.RandomBrightnessContrast(p=0.5),  # Random brightness and contrast adjustment
            A.HueSaturationValue(p=0.5),  # Change hue, saturation, and value of the image
            A.RandomShadow(p=0.5),
            A.CLAHE(p=0.6),
            A.RGBShift(p=0.8),  # Shift RGB values
            A.ChannelShuffle(p=0.8),
            A.HorizontalFlip(p=0.5),  # Random horizontal flip
            A.Rotate(limit=15),  # Random rotation in the range (-15, 15)
            A.GaussNoise(p=0.3),  # Add random noise to the image
            A.MotionBlur(blur_limit=3, p=0.3),  # Apply motion blur to the image
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)  # Keypoint's for the polygons
        )
        return augment_seq

    def data_augmentation(self, image: np.ndarray, image_labels: list) -> tuple:
        # Convert each label's polygon into key points
        key_points = [point for label in image_labels for point in label['bbox']]
        # Apply augmentations to both the image and key points
        augmented = self.transform(image=image, keypoints=key_points)
        # Extract the augmented image and key points
        augmented_image, augmented_key_points = augmented['image'], augmented['keypoints']

        augmented_image_labels, idx = [], 0
        for label in image_labels:
            num_points = len(label['bbox'])  # Get the number of points in the original polygon
            poly = augmented_key_points[idx:idx + num_points]  # Extract the augmented key points for this polygon
            idx += num_points
            # Clamp points to image boundaries after augmentation
            poly = [(min(max(0, x), image.shape[1] - 1), min(max(0, y), image.shape[0] - 1)) for x, y in poly]
            augmented_image_labels.append({'bbox': poly})
        return augmented_image, augmented_image_labels

    def __len__(self) -> int:
        return len(self.img_data)

    def __getitem__(self, idx: int) -> dict:
        image_path, image_labels = self.img_data[idx]
        image, image_height, image_width = read_image(str(image_path))
        if self.data_type == "train":
            image, image_labels = self.data_augmentation(image, image_labels)
        else:
            # the x, y coordinates will not be allowed to be out of bounds or negative to prevent errors.
            image_labels = [{'bbox': [(min(max(0, x), image_width), min(max(0, y), image_height)) for x, y in
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
    def augmentations() -> A.Compose:
        augment_seq = A.Compose([
            # Geometric augmentations
            A.Rotate(limit=3, border_mode=0, value=(0, 0, 0), p=0.8),  # Small rotations for text slant simulation
            A.Affine(scale=(0.8, 1), p=0.8),
            # Image quality changes (blur, noise)
            A.Sharpen(p=0.6),
            A.Blur(blur_limit=5, p=0.4),
            A.GaussNoise(var_limit=(50.0, 100.0), p=0.3),  # Add Gaussian noise to simulate poor-quality images
            # Color augmentations
            A.CLAHE(p=0.7),
            A.RGBShift(p=0.8),
            A.ChannelShuffle(p=0.8),
            A.PixelDropout(p=0.6),
            A.HueSaturationValue(10, sat_shift_limit=10, val_shift_limit=10, p=0.4),  # Slight color jitter
        ])
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
            image = self.transform(image=image)["image"]
        image, text = resize_norm_img(image, self.image_height, self.image_width)[0], " " if blank else text
        data = {"image_path": str(image_path), "image": image, "text": text}
        if self.preprocesses:
            for process in self.preprocesses:
                data = process(data)
                if data is None:
                    return self.__getitem__(np.random.randint(self.__len__()))
        return data
