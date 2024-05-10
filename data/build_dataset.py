import imgaug.augmenters as iaa
import numpy as np
from imgaug.augmentables import Keypoint, KeypointsOnImage
from torch.utils.data import Dataset

from data.load_data import load_data
from models.detection.db.pre_process import db_preprocess, db_collate_fn
from utilities.utils import Types, read_image, resize_norm_img, flatten_iter, pascal_voc_bb


class TextDetectionDataset(Dataset):
    def __init__(self, lang: Types.Language, data_type: Types.DataType, model_name: Types.ModelName) -> None:
        self.img_data = load_data(lang, Types.det, data_type)
        self.data_type, self.model_name, self.transform = data_type, model_name, self.augmentations()

    @staticmethod
    def augmentations() -> iaa.meta.Sequential:
        augment_seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Affine(scale=(0.7, 0.95), rotate=(-10, 10)),
            iaa.Resize((0.5, 3.0)),
            iaa.Crop(percent=(0, 0.1)),
        ])
        return augment_seq

    def data_augmentation(self, image: np.ndarray, anns: list) -> tuple:
        aug = self.transform.to_deterministic()
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

    def __len__(self) -> int:
        return len(self.img_data)

    def __getitem__(self, idx: int) -> dict:
        image_path, image_labels = self.img_data[idx]
        image = read_image(str(image_path))[0]
        if self.data_type == Types.train:
            image, image_labels = self.data_augmentation(image, image_labels)
        if self.model_name == Types.db:
            return db_preprocess(image_path, image, image_labels, 640, 640)

    def collate_fn(self, batch: list) -> dict:
        if self.model_name == Types.db:
            return db_collate_fn(batch)


class TextRecognitionDataset(Dataset):
    def __init__(self, lang: Types.Language, data_type: Types.DataType, model_name: Types.ModelName) -> None:
        self.img_data = load_data(lang, Types.rec, data_type)
        self.data_type, self.transform = data_type, self.augmentations()
        if model_name == Types.crnn:
            self.image_height, self.image_width = 32, 320

    @staticmethod
    def augmentations() -> iaa.meta.Sequential:
        augment_seq = iaa.Sequential([
            iaa.Affine(scale=(0.7, 0.95), rotate=(-10, 10)),
            iaa.Resize((0.5, 3.0)),
        ])
        return augment_seq

    def __len__(self) -> int:
        return len(self.img_data)

    def __getitem__(self, idx: int) -> dict:
        image_path, image_labels = self.img_data[idx]
        bbox = tuple(flatten_iter(image_labels[0]["bbox"]))
        x_min, y_min, x_max, y_max = map(int, pascal_voc_bb(bbox))
        image = read_image(str(image_path))[0][y_min:y_max, x_min:x_max]  # Use bbox to crop a specific text from image.
        if self.data_type == Types.train:
            image = self.transform.augment_image(image)
        image = resize_norm_img(image, self.image_height, self.image_width)[0]
        return {"image_path": str(image_path), "image": image, "text": image_labels[0]["text"]}
