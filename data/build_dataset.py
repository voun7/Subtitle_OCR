import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

from data.load_data import load_data
from utilities.utils import Types


class TextDetectionDataset(Dataset):
    def __init__(self, lang: Types.Language, data_type: Types.DataType, transform=None) -> None:
        self.img_data = load_data(lang, Types.det, data_type)
        self.img_data_keys = list(self.img_data.keys())
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_data)

    def __getitem__(self, idx: int) -> tuple:
        idx = self.img_data_keys[idx]
        img_path, bboxes = idx, self.img_data[idx]
        image = read_image(str(img_path))
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        if self.transform:
            _, orig_height, orig_width = image.shape
            image = self.transform(image)
            _, new_height, new_width = image.shape
            x_scale, y_scale = new_width / orig_width, new_height / orig_height
            bboxes[:, 0] *= x_scale  # x_min
            bboxes[:, 1] *= y_scale  # y_min
            bboxes[:, 2] *= x_scale  # x_max
            bboxes[:, 3] *= y_scale  # y_max
        return image, bboxes


class TextRecognitionDataset(Dataset):
    def __init__(self, lang: Types.Language, data_type: Types.DataType, transform=None) -> None:
        self.img_data = load_data(lang, Types.rec, data_type)
        self.img_data_keys = list(self.img_data.keys())
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_data)

    def __getitem__(self, idx: int) -> tuple:
        idx = self.img_data_keys[idx]
        img_path, texts = idx, self.img_data[idx]
        image = read_image(str(img_path))
        if self.transform:
            image = self.transform(image)
        return image, texts
