from torch.utils.data import Dataset
from torchvision.io import read_image

from data.load_data import load_data
from utilities.utils import Types


class TextDetectionDataset(Dataset):
    def __init__(self, lang: Types.Language, data_type: Types.DataType, transform=None) -> None:
        self.img_data = load_data(lang, Types.det, data_type)
        self.img_data_keys = list(self.img_data.keys())
        self.transform = transform

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        img_path, img_bboxes = self.img_data_keys[idx], self.img_data[self.img_data_keys[idx]]
        image = read_image(str(img_path))
        if self.transform:
            image = self.transform(image)
        return image, img_bboxes


class TextRecognitionDataset(Dataset):
    def __init__(self, lang: Types.Language, data_type: Types.DataType, transform=None) -> None:
        self.img_data = load_data(lang, Types.rec, data_type)
        self.img_data_keys = list(self.img_data.keys())
        self.transform = transform

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        img_path, img_texts = self.img_data_keys[idx], self.img_data[self.img_data_keys[idx]]
        image = read_image(str(img_path))
        if self.transform:
            image = self.transform(image)
        return image, img_texts
