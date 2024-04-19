import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.io import loadmat

from utilities.utils import Types, flatten_iter, pairwise_tuples
from utilities.visualize import visualize_datasource

DATASET_DIR = r"C:\Users\Victor\Documents\Python Datasets\Subtitle_OCR"


# Data Format Annotation
# sample = {
#     # image file path
#     "train_images/img_1.jpg": [
#         # ---image bbox ((x1,y1),(x2,y2),(x3,y3),(x4,y4))---    ---image texts---
#         {'bbox': ((10.0, 16.0), (50.0, 16.0), (50.0, 164.0), (10.0, 164.0)), "text": "Sample Text"},
#         {"bbox": ((1158.0, 1411.0), (1263.0, 1411.0), (1263.0, 1700.0), (1158.0, 1700.0)), "text": "Sample Text 2"},
#     ],
#     "train_images/img_100.jpg": [
#         {"bbox": ((26.0, 1659.0), (99.0, 1664.0), (88.0, 1803.0), (16.0, 1797.0)), "text": "Sample Text"},
#         {"bbox": ((0.0, 60.0), (209.0, 1.0), (229.0, 70.0), (19.0, 129.0)), "text": "Sample Text 2"},
#     ]
# }


class ChStreetViewTxtRecData:
    def __init__(self, data_type: Types.DataType) -> None:
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
        if self.data_type == Types.train:
            img_files, img_labels = train_data
        elif self.data_type == Types.val:
            img_files, img_labels = val_data
        img_data = {file: [{"bbox": None, "text": img_labels[index][0].replace("\u3000", " ")}]
                    for index, file in enumerate(img_files)}
        return img_data


class ICDAR2015Data:
    def __init__(self, data_type: Types.DataType) -> None:
        """
        ICDAR 2015 Dataset (det & rec) (en)
        source: https://rrc.cvc.uab.es/?ch=4
        """
        self.dataset_dir = Path(f"{DATASET_DIR}/ICDAR 2015/{data_type}")
        self.img_dir = self.dataset_dir / "images"
        self.labels_dir = self.dataset_dir / "labels"

    def load_data(self) -> dict:
        img_files, img_labels = list(self.img_dir.iterdir()), list(self.labels_dir.iterdir())
        img_data = {}
        for img_path, label_path in zip(img_files, img_labels):
            img_bb_txt, labels = [], label_path.read_text("utf-8-sig").splitlines()
            for label in labels:
                label = label.split(',', maxsplit=8)
                bbox = pairwise_tuples(tuple(map(float, label[:8])))
                text = label[8:][0]
                img_bb_txt.append({"bbox": bbox, "text": text})
            img_data[img_path] = img_bb_txt
        return img_data


class ICDAR2017RCTWData:
    def __init__(self, data_type: Types.DataType) -> None:
        """
        ICDAR 2017 RCTW Dataset (det & rec) (en & ch)
        source: https://rctw.vlrlab.net/dataset
        """
        self.data_type = data_type
        self.dataset_dir = Path(f"{DATASET_DIR}/ICDAR 2017 RCTW")
        self.img_dir = self.dataset_dir / "train_images"
        self.labels_dir = self.dataset_dir / "train_gts"

    def load_data(self) -> dict:
        img_files, img_labels = list(self.img_dir.iterdir()), list(self.labels_dir.iterdir())
        train_data, val_data = data_random_split(img_files, img_labels)
        if self.data_type == Types.train:
            img_files, img_labels = train_data
        elif self.data_type == Types.val:
            img_files, img_labels = val_data
        img_data = {}
        for img_path, labels in zip(img_files, img_labels):
            img_bb_txt, labels = [], labels.read_text("utf-8-sig").splitlines()
            for label in labels:
                label = label.split(',', maxsplit=9)
                bbox = pairwise_tuples(tuple(map(float, label[:8])))
                text = label[9:][0].strip('\"')
                img_bb_txt.append({"bbox": bbox, "text": text})
            img_data[img_path] = img_bb_txt
        return img_data


class ICDAR2019LSVTFullData:
    def __init__(self, data_type: Types.DataType) -> None:
        """
        ICDAR 2019 LSVT Dataset (det & rec) (en & ch)
        source: https://rrc.cvc.uab.es/?ch=16
        """
        self.data_type = data_type
        self.dataset_dir = Path(f"{DATASET_DIR}/ICDAR 2019 LSVT")
        self.img_dir = self.dataset_dir / "train_full"
        self.labels_file = self.dataset_dir / "train_full_labels.json"

    def load_img_labels(self, img_files: list) -> dict:
        with open(self.labels_file) as file:
            labels = json.load(file)
        img_data = {}
        for file in img_files:
            img_bb_txt = []
            for label in labels[file.stem]:
                bbox = label["points"]
                text = label["transcription"]
                img_bb_txt.append({"bbox": bbox, "text": text})
            img_data[file] = img_bb_txt
        return img_data

    def load_data(self) -> dict:
        img_files = list(self.img_dir.glob("*/*"))
        train_data, val_data = data_random_split(img_files)
        if self.data_type == Types.train:
            return self.load_img_labels(train_data)
        elif self.data_type == Types.val:
            return self.load_img_labels(val_data)


class ICDAR2019LSVTWeakData:
    def __init__(self, data_type: Types.DataType) -> None:
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
            img_bb_txt = []
            for label in labels[file.stem]:
                text = label["transcription"]
                img_bb_txt.append({"bbox": None, "text": text})
            img_data[file] = img_bb_txt
        return img_data

    def load_data(self) -> dict:
        img_files = list(self.img_dir.glob("*/*"))
        train_data, val_data = data_random_split(img_files)
        if self.data_type == Types.train:
            return self.load_img_labels(train_data)
        elif self.data_type == Types.val:
            return self.load_img_labels(val_data)


class SynthTextData:
    def __init__(self, data_type: Types.DataType) -> None:
        """
        SynthText Dataset (det & rec) (en)
        source: https://github.com/ankush-me/SynthText
        """
        self.data_type = data_type
        self.dataset_dir = Path(f"{DATASET_DIR}/SynthText")
        self.img_dir = self.dataset_dir / "images"
        self.labels_file = self.dataset_dir / "gt.mat"
        self.labels_json_file = self.dataset_dir / "gt.json"

    def convert_mat_to_json(self) -> None:
        labels = loadmat(str(self.labels_file))
        all_labels = {}
        for name, texts, bboxes in zip(labels["imnames"][0], labels["txt"][0], labels["wordBB"][0]):
            name = name[0].split('/')[1]
            if len(bboxes.shape) == 3:
                bboxes_t = np.transpose(bboxes)
                bboxes = bboxes_t.reshape(bboxes.shape[-1], -1).tolist()
            else:
                bboxes = np.transpose(bboxes)
                bboxes = [bboxes.flatten().tolist()]
            texts = tuple(flatten_iter([txt.split() for txt in texts]))
            assert len(bboxes) == len(texts)
            img_bb_txt = []
            for bbox, text in zip(bboxes, texts):
                img_bb_txt.append({"bbox": pairwise_tuples(bbox), "text": text})
            all_labels[name] = img_bb_txt
        with open(self.labels_json_file, "w") as outfile:
            json.dump(all_labels, outfile)

    def load_img_labels(self, img_files: list) -> dict:
        if not self.labels_json_file.exists():
            self.convert_mat_to_json()

        with open(self.labels_json_file) as file:
            labels = json.load(file)

        return {file: labels[file.name] for file in img_files}

    def load_data(self) -> dict:
        img_files = list(self.img_dir.glob("*/*"))
        train_data, val_data = data_random_split(img_files)
        if self.data_type == Types.train:
            return self.load_img_labels(train_data)
        elif self.data_type == Types.val:
            return self.load_img_labels(val_data)


class TextOCR01Data:
    def __init__(self, data_type: Types.DataType) -> None:
        """
        TextOCR v0.1 Dataset (det & rec) (en)
        source: https://textvqa.org/textocr/dataset/
        """
        self.dataset_dir = Path(f"{DATASET_DIR}/TextOCR V0.1")
        self.img_dir = self.dataset_dir / "train_val_images"
        self.labels_file = self.dataset_dir / f"TextOCR_0.1_{data_type}.json"

    def load_labels(self) -> dict:
        with open(self.labels_file) as file:
            labels = json.load(file)["anns"].values()

        all_labels = {}
        for val in labels:
            text = val["utf8_string"]
            if len(text) > 1:
                img_id = val["id"].split('_')[0]
                bbox = pairwise_tuples(val["points"])
                all_labels.setdefault(img_id, []).append({"bbox": bbox, "text": text})
        return all_labels

    def load_data(self) -> dict:
        img_labels = self.load_labels()
        img_files = {file.stem: file for file in self.img_dir.iterdir()}
        img_data = {img_files[key]: value for key, value in img_labels.items()}
        return img_data


class TRDGSyntheticData:
    def __init__(self, lang: Types.Language, data_type: Types.DataType) -> None:
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
        img_data = {img_file: [{"bbox": None, "text": labels[img_file.name]}] for img_file in img_files}
        return img_data

    def load_data(self) -> dict:
        img_files = list(self.img_dir.glob("*.jpg"))
        train_data, val_data = data_random_split(img_files)
        if self.data_type == Types.train:
            return self.load_img_labels(train_data)
        elif self.data_type == Types.val:
            return self.load_img_labels(val_data)


def data_random_split(files, labels=None, split: float = 0.8, shuffle: bool = False) -> tuple[np.array, np.array]:
    """
    Randomly split any given files into two sets of non-overlapping training and validation files.
    :param files: Index able object with length.
    :param labels: Labels for the files. Files and labels should have the same sort if provided.
    :param split: Represents the ratio proportion of the dataset split for training and validation.
    :param shuffle: Value used to set randomness of split. For reproducibility purposes.
    """
    files = np.array(files)
    files_size = len(files)
    train_size = int(split * files_size)

    seed = None if shuffle else 31
    rng = np.random.default_rng(seed)
    random_indexes = rng.permutation(files_size)

    train_indexes, val_indexes = random_indexes[:train_size], random_indexes[train_size:]
    train_files, val_files = files[train_indexes], files[val_indexes]
    if labels:
        labels = np.array(labels)
        assert len(labels) == files_size
        train_labels, val_labels = labels[train_indexes], labels[val_indexes]
        return (train_files, train_labels), (val_files, val_labels)
    return train_files, val_files


def merge_data_sources(*args) -> dict:
    all_images_data = {}
    for ds in args:
        img_data = ds.load_data()
        all_images_data.update(img_data)
    return all_images_data


def load_data(lang: Types.Language, model_type: Types.ModelType, data_type: Types.DataType) -> dict:
    if lang == Types.english:
        if model_type == Types.det:
            return merge_data_sources(
                ICDAR2015Data(data_type),
                # SynthTextData(data_type),
                # TextOCR01Data(data_type),
            )
        elif model_type == Types.rec:
            return merge_data_sources(
                ICDAR2015Data(data_type),
                SynthTextData(data_type),
                TextOCR01Data(data_type),
                TRDGSyntheticData(lang, data_type)
            )
    elif lang == Types.chinese:
        if model_type == Types.det:
            return merge_data_sources(
                ICDAR2017RCTWData(data_type),
                ICDAR2019LSVTFullData(data_type)
            )
        elif model_type == Types.rec:
            return merge_data_sources(
                ChStreetViewTxtRecData(data_type),
                ICDAR2017RCTWData(data_type),
                ICDAR2019LSVTFullData(data_type),
                ICDAR2019LSVTWeakData(data_type),
                TRDGSyntheticData(lang, data_type)
            )


if __name__ == '__main__':
    start = perf_counter()

    ts_data = load_data(Types.english, Types.det, Types.train)
    ts_keys = list(ts_data.keys())
    ts_len = len(ts_keys)
    print(f"Data Source Length: {ts_len:,} Data Load Time: {perf_counter() - start:.4f}\n")
    for ts_idx in range(0, ts_len, round(ts_len / 200)):
        ts_img_path, ts_img_labels = str(ts_keys[ts_idx]), ts_data[ts_keys[ts_idx]]
        print(f"Image Path: {ts_img_path}\nImage Labels: {ts_img_labels}\n")
        visualize_datasource(ts_img_path, ts_img_labels)
