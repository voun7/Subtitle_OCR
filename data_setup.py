import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.io import read_image


class TRDGDataset(Dataset):
    def __init__(self, image_dir: str) -> None:
        self.image_dir = Path(image_dir)
        self.image_files, self.image_texts, self.bboxes = self.load_data()

    def __getitem__(self, index: int) -> tuple:
        image_path, image_text, image_bboxes = self.image_files[index], self.image_texts[index], self.bboxes[index]
        image = read_image(str(image_path))
        return image, image_text, image_bboxes

    def __len__(self) -> int:
        return len(self.image_files)

    def load_data(self) -> tuple:
        image_files, labels, bboxes = [], [], []
        image_ext = ".jpg"
        for image_file in self.image_dir.glob(f"*{image_ext}"):
            image_files.append(image_file)
            image_txt = image_file.stem.split('_')[0]
            labels.append(image_txt)
            bbox_file = str(image_file).replace(image_ext, '_boxes.txt')
            bbox_data = self.load_bounding_boxes(bbox_file)
            bboxes.append(bbox_data)
        return image_files, labels, bboxes

    @staticmethod
    def load_bounding_boxes(bbox_file: str) -> list:
        bbox_file_lines = Path(bbox_file).read_text().splitlines()
        bboxes = []
        for line in bbox_file_lines:
            values = line.split()
            x1, y1, x2, y2 = map(int, values)
            bbox = [x1, y1, x2, y2]  # Format: top left (x1, y1) and bottom right (x2, y2)
            bboxes.append(bbox)
        return bboxes


def collate_fn(batch):
    """
    The collate_fn parameter in the DataLoader allows you to specify a function
    that controls the behavior of batching samples from your dataset.
    """
    image, labels, bboxes = zip(*batch)
    # Stack images, labels, and bboxes into batches
    image = torch.stack(image)
    return image, labels, bboxes


def create_dataloaders(image_dir: str, batch_size: int, split_ratio: float = 0.8, shuffle: bool = True) \
        -> tuple[DataLoader, DataLoader]:
    # Create an instance of the TRDGDataset.
    dataset = TRDGDataset(image_dir)
    # Perform train-test split.
    dataset_size = len(dataset)
    train_size = int(split_ratio * dataset_size)
    test_size = dataset_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    num_workers = os.cpu_count()
    # Create the data loaders for training and testing.
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn
    )
    return train_dataloader, test_dataloader
