import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.io import loadmat

from utilities.visualize import visualize_datasource

DATASET_DIR = r"C:\Users\Victor\Documents\Python Datasets\Subtitle_OCR"


# Text Detection Format Annotation
# -Image file path--          --Image bboxes (top left & bottom right)--
# {"train_images/img_1.jpg": [(310, 104, 416, 141), ..., (310, 104, 416, 141)]}

# Text Recognition Format Annotation
# --Image file path--         --Image text--
# {"train_images/img_1.jpg": ["Sample Text" ..., "Sample Text2"]}


class ChStreetViewTxtRecData:
    def __init__(self, data_type: str) -> None:
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
    def __init__(self, model_type: str, data_type: str) -> None:
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
    def __init__(self, model_type: str, data_type: str) -> None:
        """
        ICDAR 2019 LSVT Dataset (det & rec) (en & ch)
        source: https://rrc.cvc.uab.es/?ch=16
        """
        self.model_type, self.data_type = model_type, data_type
        self.dataset_dir = Path(f"{DATASET_DIR}/ICDAR 2019 LSVT")
        self.img_dir = self.dataset_dir / "train_full"
        self.labels_file = self.dataset_dir / "train_full_labels.json"

    def load_img_labels(self, img_files: list) -> dict:
        with open(self.labels_file) as file:
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
    def __init__(self, data_type: str) -> None:
        """
        ICDAR 2019 LSVT Dataset (rec) (en & ch)
        source: https://rrc.cvc.uab.es/?ch=16
        """
        self.data_type = data_type
        self.dataset_dir = Path(f"{DATASET_DIR}/ICDAR 2019 LSVT")
        self.img_dir = self.dataset_dir / "train_weak"
        self.labels_file = self.dataset_dir / "train_weak_labels.json"

    def load_img_labels(self, img_files: list) -> dict:
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
    def __init__(self, model_type: str, data_type: str) -> None:
        """
        SynthText Dataset (det & rec) (en)
        source: https://github.com/ankush-me/SynthText
        """
        self.model_type, self.data_type = model_type, data_type
        self.dataset_dir = Path(f"{DATASET_DIR}/SynthText")
        self.img_dir = self.dataset_dir / "images"
        self.labels_file = self.dataset_dir / "gt.mat"

    def load_img_labels(self, img_files: list) -> dict:
        labels = loadmat(str(self.labels_file))

        img_data = {}
        if self.model_type == "det":
            all_labels = {name[0].split('/')[1]: bbs for name, bbs in zip(labels["imnames"][0], labels["wordBB"][0])}
            for file in img_files:
                bboxes = all_labels[file.name]
                if len(bboxes.shape) == 3:
                    bboxes_t = np.transpose(bboxes)
                    reshaped_bboxes = bboxes_t.reshape(bboxes.shape[-1], -1)
                    img_data[file] = reshaped_bboxes[:, [0, 1, 4, 5]]
                else:
                    img_data[file] = [(bboxes[0][0], bboxes[1][0], bboxes[0][2], bboxes[1][2])]
        elif self.model_type == "rec":
            all_labels = {name[0].split('/')[1]: texts for name, texts in zip(labels["imnames"][0], labels["txt"][0])}
            img_data = {file: all_labels[file.name] for file in img_files}
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
    def __init__(self, model_type: str, data_type: str) -> None:
        """
        TextOCR v0.1 Dataset (det & rec) (en)
        source: https://textvqa.org/textocr/dataset/
        """
        self.model_type = model_type
        self.dataset_dir = Path(f"{DATASET_DIR}/TextOCR V0.1")
        self.img_dir = self.dataset_dir / "train_val_images"
        self.labels_file = self.dataset_dir / f"TextOCR_0.1_{data_type}.json"

    def load_labels(self) -> dict:
        with open(self.labels_file) as file:
            labels = json.load(file)["anns"].values()

        all_labels = {}
        if self.model_type == "det":
            for val in labels:
                text = val["utf8_string"]
                if len(text) > 1:
                    img_id = val["id"].split('_')[0]
                    bbox = val["points"]
                    bbox = bbox[0], bbox[1], bbox[4], bbox[5]
                    all_labels.setdefault(img_id, []).append(bbox)
        elif self.model_type == "rec":
            for val in labels:
                text = val["utf8_string"]
                if len(text) > 1:
                    img_id = val["id"].split('_')[0]
                    all_labels.setdefault(img_id, []).append(text)
        return all_labels

    def load_data(self) -> dict:
        img_labels = self.load_labels()
        img_files = {file.stem: file for file in self.img_dir.iterdir()}
        img_data = {img_files[key]: value for key, value in img_labels.items()}
        return img_data


class TRDGSyntheticData:
    def __init__(self, lang: str, data_type: str) -> None:
        """
        Multilingual Dataset made using trdg package. (rec) (multi lang)
        Language depends on what data was generated.
        source: https://github.com/Belval/TextRecognitionDataGenerator
        """
        self.data_type = data_type
        self.dataset_dir = Path(f"{DATASET_DIR}/TRDG Synthetic Images")
        self.img_dir = self.dataset_dir / lang
        self.labels_file = self.img_dir / f"labels.txt"

    def load_img_labels(self, img_files: list) -> dict:
        labels = {text.split(maxsplit=1)[0]: text.split(maxsplit=1)[1] for text in
                  self.labels_file.read_text("utf-8").splitlines()}
        img_data = {img_file: labels[img_file.name] for img_file in img_files}
        return img_data

    def load_data(self) -> dict:
        img_files = list(self.img_dir.glob("*.jpg"))
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
                TextOCR01Data(model_type, data_type)
            )
        elif model_type == "rec":
            return merge_data_sources(
                SynthTextData(model_type, data_type),
                TextOCR01Data(model_type, data_type),
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


if __name__ == '__main__':
    start = perf_counter()

    tds = TRDGSyntheticData("en", "train")
    ts_data = tds.load_data()
    ts_keys, ts_idx = list(ts_data.keys()), 0
    ts_img_path, ts_img_labels = str(ts_keys[ts_idx]), ts_data[ts_keys[ts_idx]]
    print(f"Data Source Length: {len(ts_keys):,}\nImage Path: {ts_img_path}\nImage Labels: {ts_img_labels}\n"
          f"Data Load Time: {(perf_counter() - start):.4f}")
    if "str" in str(type(ts_img_labels[0])):  # Check 1st value. Labels that contain texts will be strings.
        visualize_datasource(ts_img_path)
    else:
        visualize_datasource(ts_img_path, bboxes=ts_img_labels)
