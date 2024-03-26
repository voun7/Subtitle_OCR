from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2

from data.load_data import load_data


class TextDetectionDataset(Dataset):
    def __init__(self, lang: str, data_type: str, transform: v2 = None) -> None:
        self.img_paths, self.img_targets = load_data(lang, "det", data_type)
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = read_image(img_path)
        target = self.img_targets[idx]
        if self.transform:
            image = self.transform(image)
        return image, target


class TextRecognitionDataset(Dataset):
    def __init__(self, lang: str, data_type: str, transform: v2 = None) -> None:
        self.img_paths, self.img_labels = load_data(lang, "rec", data_type)
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = read_image(img_path)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
