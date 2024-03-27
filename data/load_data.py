import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

DATASET_DIR = r"C:\Users\Victor\Documents\Python Datasets\Subtitle_OCR"


# Text Detection Format Annotation
# -Image file path--          --Image bboxes (top left & bottom right)--
# {"train_images/img_1.jpg": [[310, 104, 416, 141], ..., [310, 104, 416, 141]]}

# Text Recognition Format Annotation
# --Image file path--         --Image text--
# {"train_images/img_1.jpg": ["Sample Text" ..., "Sample Text2"]}


class ChStreetViewTxtRecData:
    def __init__(self, data_type: str = "train") -> None:
        """
        Chinese Scene Text Recognition Dataset (rec) (ch)
        source: https://aistudio.baidu.com/competition/detail/8/0/related-material
        """
        self.data_type = data_type
        self.dataset_dir = Path(f"{DATASET_DIR}/Chinese Street View Text Recognition")
        self.img_dir = self.dataset_dir / "train"
        self.labels_file = self.dataset_dir / "train.txt"

    def load_data(self) -> dict:
        img_files = sorted(self.img_dir.iterdir(), key=lambda name: float(name.stem.split("_")[-1]))
        img_labels = [label.split(maxsplit=3)[3:] for label in self.labels_file.read_text("utf-8").splitlines()]

        train_data, val_data = data_random_split(img_files, img_labels)
        if self.data_type == "train":
            img_files, img_labels = train_data
        elif self.data_type == "val":
            img_files, img_labels = val_data
        img_data = {file: img_labels[index] for index, file in enumerate(img_files)}
        return img_data


class ICDAR2017RCTWData:
    def __init__(self, model_type, data_type: str = "train") -> None:
        """
        ICDAR 2017 RCTW Dataset (det & rec) (en & ch)
        source: https://rctw.vlrlab.net/dataset
        """
        self.model_type, self.data_type = model_type, data_type
        self.dataset_dir = Path(f"{DATASET_DIR}/ICDAR 2017 RCTW")
        self.img_dir = self.dataset_dir / "train_images"
        self.labels_dir = self.dataset_dir / "train_gts"

    def read_labels(self, label_files: list) -> list:
        all_labels = []
        if self.model_type == "det":
            for file in label_files:
                boxes, text = [], file.read_text("utf-8-sig").splitlines()
                for labels in text:
                    labels = labels.split(',', maxsplit=9)
                    bboxes = int(labels[:7][0]), int(labels[:7][1]), int(labels[:7][4]), int(labels[:7][5])
                    boxes.append(bboxes)
                all_labels.append(boxes)
        elif self.model_type == "rec":
            for file in label_files:
                texts, text = [], file.read_text("utf-8").splitlines()
                for labels in text:
                    labels = labels.split(',', maxsplit=9)
                    text = labels[9:][0].strip('\"')
                    texts.append(text)
                all_labels.append(texts)
        return all_labels

    def load_data(self) -> dict:
        img_files, img_labels = list(self.img_dir.iterdir()), list(self.labels_dir.iterdir())
        train_data, val_data = data_random_split(img_files, img_labels)
        if self.data_type == "train":
            img_files, img_labels = train_data
        elif self.data_type == "val":
            img_files, img_labels = val_data
        img_labels = self.read_labels(img_labels)
        img_data = {file: img_labels[index] for index, file in enumerate(img_files)}
        return img_data


class ICDAR2019LSVTFullData:
    def __init__(self, model_type, data_type: str = "train") -> None:
        """
        ICDAR 2019 LSVT Dataset (det & rec) (en & ch)
        source: https://rrc.cvc.uab.es/?ch=16
        """
        self.model_type, self.data_type = model_type, data_type
        self.dataset_dir = Path(f"{DATASET_DIR}/ICDAR 2019 LSVT")
        self.img_dir = self.dataset_dir / "train_full"
        self.labels_file = self.dataset_dir / "train_full_labels.json"

    def load_img_labels(self, img_files) -> dict:
        with open(self.labels_file, 'r') as file:
            labels = json.load(file)
        img_data = {}
        if self.model_type == "det":
            for file in img_files:
                boxes = []
                for label in labels[file.stem]:
                    bboxes = label["points"]
                    bboxes = bboxes[0][0], bboxes[0][1], bboxes[2][0], bboxes[2][1]
                    boxes.append(bboxes)
                img_data[file] = boxes
        elif self.model_type == "rec":
            for file in img_files:
                texts = []
                for label in labels[file.stem]:
                    text = label["transcription"]
                    texts.append(text)
                img_data[file] = texts
        return img_data

    def load_data(self) -> dict:
        img_files = list(self.img_dir.glob("*/*"))
        train_data, val_data = data_random_split(img_files)
        if self.data_type == "train":
            img_data = self.load_img_labels(train_data)
            return img_data
        elif self.data_type == "val":
            img_data = self.load_img_labels(val_data)
            return img_data


class ICDAR2019LSVTWeakData:
    def __init__(self, data_type: str = "train") -> None:
        """
        ICDAR 2019 LSVT Dataset (rec) (en & ch)
        source: https://rrc.cvc.uab.es/?ch=16
        """
        self.data_type = data_type
        self.dataset_dir = Path(f"{DATASET_DIR}/ICDAR 2019 LSVT")
        self.img_dir = self.dataset_dir / "train_weak"
        self.labels_file = self.dataset_dir / "train_weak_labels.json"

    def load_img_labels(self, img_files) -> dict:
        with open(self.labels_file, 'r') as file:
            labels = json.load(file)
        img_data = {}
        for file in img_files:
            texts = []
            for label in labels[file.stem]:
                text = label["transcription"]
                texts.append(text)
            img_data[file] = texts
        return img_data

    def load_data(self) -> dict:
        img_files = list(self.img_dir.glob("*/*"))
        train_data, val_data = data_random_split(img_files)
        if self.data_type == "train":
            img_data = self.load_img_labels(train_data)
            return img_data
        elif self.data_type == "val":
            img_data = self.load_img_labels(val_data)
            return img_data


class SynthTextData:
    def __init__(self, model_type, data_type: str = "train") -> None:
        """
        SynthText Dataset (det & rec) (en)
        source: https://github.com/ankush-me/SynthText
        """
        self.model_type, self.data_type = model_type, data_type
        self.dataset_dir = Path(f"{DATASET_DIR}/SynthText")
        self.img_dir = self.dataset_dir / "images"
        self.labels_file = self.dataset_dir / "gt.mat"

    def load_img_labels(self, img_files) -> dict:
        # TODO: Make loading of labels faster.
        labels = loadmat(str(self.labels_file))
        file_names = np.array([file_name[0] for file_name in labels["imnames"][0]])
        img_data = {}
        if self.model_type == "det":
            img_bboxes = labels["wordBB"][0]
            for file in img_files:
                boxes, ann_key = [], '/'.join(file.parts[-2:])
                index = np.where(file_names == np.array(ann_key))[0][0]
                for bbox in img_bboxes[index]:
                    # TODO: Append correct bbox coordinates.
                    # bbox = bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1]
                    boxes.append(bbox)
                img_data[file] = boxes
        elif self.model_type == "rec":
            img_text = labels["txt"][0]
            for file in img_files:
                texts, ann_key = [], '/'.join(file.parts[-2:])
                index = np.where(file_names == ann_key)[0][0]
                for text in img_text[index]:
                    texts.append(text)
                img_data[file] = texts
        return img_data

    def load_data(self) -> dict:
        img_files = list(self.img_dir.glob("*/*"))
        train_data, val_data = data_random_split(img_files)
        if self.data_type == "train":
            img_data = self.load_img_labels(train_data)
            return img_data
        elif self.data_type == "val":
            img_data = self.load_img_labels(val_data)
            return img_data


class TextOCR01Data:
    def __init__(self, data_type: str = "train") -> None:
        """
        TextOCR v0.1 Dataset (det) (en)
        source: https://textvqa.org/textocr/dataset/
        """
        self.dataset_dir = Path(f"{DATASET_DIR}/TextOCR V0.1")
        self.img_dir = self.dataset_dir / "train_val_images"
        self.labels_file = self.dataset_dir / f"TextOCR_0.1_{data_type}.json"

    def load_labels(self) -> dict:
        with open(self.labels_file) as file:
            labels = json.load(file)["anns"]

        all_labels = {}
        for val in labels.values():
            img_id = val["id"].split('_')[0]
            bbox = val["bbox"]
            all_labels.setdefault(img_id, []).append(bbox)
        return all_labels

    def load_data(self) -> dict:
        img_labels = self.load_labels()
        label_keys, img_data = set(img_labels.keys()), {}
        for img_file in self.img_dir.iterdir():
            if img_file.stem in label_keys:
                img_data[img_file] = img_labels[img_file.stem]
        return img_data


class TRDGSyntheticData:
    def __init__(self, lang: str, data_type: str = "train") -> None:
        """
        Multilingual Dataset made using trdg package. (rec) (multi lang)
        Language depends on what data was generated.
        source: https://github.com/Belval/TextRecognitionDataGenerator
        """
        self.data_type = data_type
        self.dataset_dir = Path(f"{DATASET_DIR}/TRDG Synthetic Images/{lang}")
        self.img_dir = self.dataset_dir / "train_val_images"
        self.labels_file = self.dataset_dir / f"train_val_labels.json"

    def load_img_labels(self, img_files) -> dict:
        pass

    def load_data(self) -> dict:
        img_files = list(self.img_dir.iterdir())
        train_data, val_data = data_random_split(img_files)
        if self.data_type == "train":
            img_data = self.load_img_labels(train_data)
            return img_data
        elif self.data_type == "val":
            img_data = self.load_img_labels(val_data)
            return img_data


def data_random_split(files, labels=None, split: float = 0.8, seed: int = 31) -> tuple[np.array, np.array]:
    """
    Randomly split any given files into two sets of non-overlapping training and validation files.
    :param files: Index able object with length.
    :param labels: labels for the files. files and labels should have the same sort if provided.
    :param split: represent the proportion of the dataset to include in the test split
    :param seed: Value used to set randomness of split. For reproducibility purposes.
    """
    files = np.array(files)
    files_size = len(files)
    train_size = int(split * files_size)

    rng = np.random.default_rng(seed)
    random_indexes = rng.permutation(files_size)

    train_indexes, val_indexes = random_indexes[:train_size], random_indexes[train_size:]
    train_files, val_files = files[train_indexes], files[val_indexes]
    if labels:
        labels = np.array(labels)
        assert len(labels) == files_size
        train_labels, val_labels = labels[train_indexes], labels[val_indexes]
        return (train_files, train_labels), (train_files, train_labels)
    return train_files, val_files


def merge_data_sources(*args) -> dict:
    all_images_data = {}
    for ds in args:
        img_data = ds.load_data()
        all_images_data.update(img_data)
    return all_images_data


def load_data(lang: str, model_type: str, data_type: str) -> dict:
    if lang == "en":
        if model_type == "det":
            return merge_data_sources(
                SynthTextData(model_type, data_type),
                TextOCR01Data(data_type)
            )
        elif model_type == "rec":
            return merge_data_sources(
                SynthTextData(model_type, data_type),
                TRDGSyntheticData(lang, data_type)
            )
    elif lang == "ch":
        if model_type == "det":
            return merge_data_sources(
                ICDAR2017RCTWData(model_type, data_type),
                ICDAR2019LSVTFullData(model_type, data_type)
            )
        elif model_type == "rec":
            return merge_data_sources(
                ChStreetViewTxtRecData(data_type),
                ICDAR2017RCTWData(model_type, data_type),
                ICDAR2019LSVTFullData(model_type, data_type),
                ICDAR2019LSVTWeakData(data_type),
                TRDGSyntheticData(lang, data_type)
            )
