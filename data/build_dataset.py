from torch.utils.data import Dataset
from torchvision.io import read_image

from data.load_data import load_data
from data.pre_processes.db_processes import db_preprocess, db_collate_fn
from utilities.utils import Types


class TextDetectionDataset(Dataset):
    def __init__(self, lang: Types.Language, data_type: Types.DataType, model_name: Types.ModelName) -> None:
        self.img_data = load_data(lang, Types.det, data_type)
        self.img_data_keys = list(self.img_data.keys())
        self.data_type, self.model_name = data_type, model_name

    def __len__(self) -> int:
        return len(self.img_data)

    def __getitem__(self, idx: int) -> dict:
        idx = self.img_data_keys[idx]
        image_path, img_labels = idx, self.img_data[idx]
        if self.model_name == Types.db:
            return db_preprocess(str(image_path), img_labels, self.data_type)

    def collate_fn(self, batch: list) -> dict:
        if self.model_name == Types.db:
            return db_collate_fn(batch)


class TextRecognitionDataset(Dataset):
    def __init__(self, lang: Types.Language, data_type: Types.DataType, model_name: Types.ModelName,
                 transform=None) -> None:
        self.img_data = load_data(lang, Types.rec, data_type)
        self.img_data_keys = list(self.img_data.keys())
        self.model_name, self.transform = model_name, transform

    def __len__(self) -> int:
        return len(self.img_data)

    def __getitem__(self, idx: int) -> tuple:
        idx = self.img_data_keys[idx]
        image_path, img_labels = idx, self.img_data[idx]
        image = read_image(str(image_path))
        if self.transform:
            image = self.transform(image)
        return image, img_labels["text"]

    @staticmethod
    def collate_fn(batch: list) -> tuple:
        """
        If your dataset contains samples with varying sizes (e.g. images with different numbers of texts),
        you need a collate function to properly batch them together.
        """
        return tuple(zip(*batch))
