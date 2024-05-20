import imgaug.augmenters as iaa
import numpy as np
from imgaug.augmentables import Keypoint, KeypointsOnImage
from torch.utils.data import Dataset

from data.load_data import load_data
from models.detection.db.pre_process import db_preprocess, db_collate_fn
from utilities.utils import Types, read_image, resize_norm_img, rescale, flatten_iter, pairwise_tuples, crop_image


class TextDetectionDataset(Dataset):
    def __init__(self, lang: Types.Language, data_type: Types.DataType, model_name: Types.ModelName, image_height: int,
                 image_width: int) -> None:
        self.img_data = load_data(lang, Types.det, data_type)
        self.data_type, self.model_name, self.transform = data_type, model_name, self.augmentations()
        self.image_height, self.image_width = image_height, image_width

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
        if self.data_type == Types.train:
            image, image_labels = self.data_augmentation(image, image_labels)
        else:
            # the bbox coordinates will not be allowed to be out of bounds or negative to prevent errors.
            image_labels = [{'bbox': [(max(min(x, image_width), 1), max(min(y, image_height), 1)) for x, y in
                                      label["bbox"]]} for label in image_labels]
        image, scale = resize_norm_img(image, self.image_height, self.image_width)
        image_labels = [{"bbox": pairwise_tuples(rescale(scale, bbox=tuple(flatten_iter(label["bbox"]))))} for label in
                        image_labels]
        if self.model_name == Types.db:
            return db_preprocess(image_path, image, image_labels, self.image_height, self.image_width)

    def collate_fn(self, batch: list) -> dict:
        if self.model_name == Types.db:
            return db_collate_fn(batch)


class TextRecognitionDataset(Dataset):
    def __init__(self, lang: Types.Language, data_type: Types.DataType, model_name: Types.ModelName, image_height: int,
                 image_width: int) -> None:
        self.img_data = load_data(lang, Types.rec, data_type)
        self.data_type, self.model_name, self.transform = data_type, model_name, self.augmentations()
        self.image_height, self.image_width = image_height, image_width

    @staticmethod
    def augmentations() -> iaa.meta.Sequential:
        augment_seq = iaa.Sequential([
            iaa.Affine(scale=(0.8, 1.2), rotate=(-5, 5)),
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
        if self.data_type == Types.train:
            image = self.transform.augment_image(image)
        image = resize_norm_img(image, self.image_height, self.image_width)[0]
        return {"image_path": str(image_path), "image": image, "text": " " if blank else text}


def test_dataset() -> None:
    """
    Code for testing index's of the dataset for errors.
    """
    dataset = TextDetectionDataset(Types.english, Types.train, Types.db, 640, 640)
    # dataset = TextRecognitionDataset(Types.english, Types.train, Types.crnn, 32, 160)
    # print(dataset[])
    dataset_len, start_idx = len(dataset), 0
    print(f"Dataset Size: {dataset_len:,}")
    for idx in range(start_idx, dataset_len):
        try:
            _ = dataset[idx]
            print(f"\ridx: {idx:,}/{dataset_len:,} passed test", end='', flush=True)
        except Exception as error:
            print(end="\r", flush=True)
            print(f"idx {idx} failed test. {error}")


if __name__ == '__main__':
    test_dataset()
