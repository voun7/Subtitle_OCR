from torch.utils.data import Dataset
from torchvision import tv_tensors
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
        img_path, img_labels = idx, self.img_data[idx]
        image = read_image(str(img_path))
        image = tv_tensors.Image(image)
        orig_height, orig_width = image.shape[-2:]
        # bboxes = tv_tensors.BoundingBoxes(bboxes, format="XYXY", canvas_size=(orig_height, orig_width))
        bboxes = img_labels["bbox"]
        if self.transform:
            image, bboxes = self.transform(image, bboxes)
        return image, bboxes

    @staticmethod
    def collate_fn(batch: list) -> tuple:
        """
        If your dataset contains samples with varying sizes (e.g. images with different numbers of bounding boxes),
        you need a collate function to properly batch them together.
        """
        return tuple(zip(*batch))


class TextRecognitionDataset(Dataset):
    def __init__(self, lang: Types.Language, data_type: Types.DataType, transform=None) -> None:
        self.img_data = load_data(lang, Types.rec, data_type)
        self.img_data_keys = list(self.img_data.keys())
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_data)

    def __getitem__(self, idx: int) -> tuple:
        idx = self.img_data_keys[idx]
        img_path, img_labels = idx, self.img_data[idx]
        image = read_image(str(img_path))
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
