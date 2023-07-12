# A file to prepare and download data if needed.
import os
from pathlib import Path

from torch.utils.data import DataLoader, random_split
from torchvision.datasets.folder import default_loader
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import ToTensor


class TRDGDataset(VisionDataset):
    def __init__(self, root: str, transform=None, target_transform=None) -> None:
        super().__init__(root, transform, target_transform)
        self.data_dir = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = default_loader  # Responsible for loading an image from the given path and returning it as a PIL.
        self.image_files, self.labels, self.bboxes = self.load_data()

    def __getitem__(self, index: int) -> tuple:
        path, target, bbox = self.image_files[index], self.labels[index], self.bboxes[index]
        image = self.loader(str(path))
        if self.transform is not None:
            image = self.transform(image)
        return image, target, bbox

    def __len__(self) -> int:
        return len(self.image_files)

    def load_data(self) -> tuple:
        image_files, labels, bboxes = [], [], []
        label_file = self.data_dir / "labels.txt"
        label_lines = label_file.read_text(encoding="utf-8").splitlines()

        for line in label_lines:
            line = line.split()
            image_file = self.data_dir / line[0]
            if image_file.exists():
                image_files.append(image_file)
                labels.append(line[1])
                bbox_file = str(image_file).replace('.jpg', '_boxes.txt')
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


def create_dataloaders(data_dir: str, batch_size: int, split_ratio: float = 0.8, shuffle: bool = True) -> tuple:
    # Create an instance of the TRDGDataset.
    dataset = TRDGDataset(data_dir, transform=ToTensor())
    # Perform train-test split.
    dataset_size = len(dataset)
    train_size = int(split_ratio * dataset_size)
    test_size = dataset_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    num_workers = os.cpu_count()
    # Create the data loaders for training and testing.
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return train_dataloader, test_dataloader
