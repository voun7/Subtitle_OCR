from torch.utils.data import Dataset

from data.load_data import load_data
from data.pre_processes.crnn_processes import crnn_preprocess, crnn_collate_fn
from data.pre_processes.db_processes import db_preprocess, db_collate_fn
from utilities.utils import Types


class TextDetectionDataset(Dataset):
    def __init__(self, lang: Types.Language, data_type: Types.DataType, model_name: Types.ModelName) -> None:
        self.img_data = load_data(lang, Types.det, data_type)
        self.data_type, self.model_name = data_type, model_name

    def __len__(self) -> int:
        return len(self.img_data)

    def __getitem__(self, idx: int) -> dict:
        image_path, img_labels = self.img_data[idx]
        if self.model_name == Types.db:
            return db_preprocess(str(image_path), img_labels, self.data_type)

    def collate_fn(self, batch: list) -> dict:
        if self.model_name == Types.db:
            return db_collate_fn(batch)


class TextRecognitionDataset(Dataset):
    def __init__(self, lang: Types.Language, data_type: Types.DataType, model_name: Types.ModelName) -> None:
        self.img_data = load_data(lang, Types.rec, data_type)
        self.data_type, self.model_name = data_type, model_name

    def __len__(self) -> int:
        return len(self.img_data)

    def __getitem__(self, idx: int) -> dict:
        image_path, img_labels = self.img_data[idx]
        if self.model_name == Types.crnn:
            return crnn_preprocess(str(image_path), img_labels, self.data_type)

    def collate_fn(self, batch: list) -> dict:
        if self.model_name == Types.crnn:
            return crnn_collate_fn(batch)
