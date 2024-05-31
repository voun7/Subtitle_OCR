import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.io import loadmat

from utilities.utils import Types, flatten_iter, pairwise_tuples, rect_corners
from utilities.visualize import visualize_data

DATASET_DIR = Path(r"C:\Users\Victor\Documents\Python Datasets\Subtitle_OCR")


# Data Format Annotation
# sample = [
#     # image file path
#     ("images/img_1.jpg", [
#         # ---image bbox ((x1,y1),(x2,y2),(x3,y3),(x4,y4))---                 ---image texts---
#         {"bbox": ((10.0, 16.0), (50.0, 16.0), (50.0, 16), (10.0, 164.0)), "text": "Sample Text"},
#         {"bbox": ((118.0, 11.0), (13.0, 141.0), (13, 170.0), (18, 10)), "text": "Sample Text 2"}
#     ]),
#     # recognition data will have only one value in list
#     ("images/img_1.jpg", [{"bbox": ((10.0, 16.0), (50.0, 16.0), (50.0, 16), (10.0, 164.0)), "text": "Sample Text"}]),
#     ("images/img_1.jpg", [{"bbox": ((118.0, 11.0), (13.0, 141.0), (13, 170.0), (18, 10)), "text": "Sample Text 2"}]),
#     ("images/img_100.jpg", [{"bbox": ((26.0, 16.0), (99.0, 14), (88.0, 80.0), (16.0, 17)), "text": "Sample Text 5"}]),
#     ("images/img_1000.jpg", [{"bbox": None, "text": "Sample Text 6"}])
# ]


class ChStreetViewTxtRecData:
    def __init__(self, data_type: Types.DataType) -> None:
        """
        Chinese Scene Text Recognition Dataset (rec) (ch) (Word & Line)
        source: https://aistudio.baidu.com/competition/detail/8/0/related-material
        """
        self.data_type, self.dataset_dir = data_type, DATASET_DIR / "Chinese Street View Text Recognition"
        self.image_dir, self.labels_file = self.dataset_dir / "train", self.dataset_dir / "train.txt"

    def load_data(self) -> list:
        image_files = sorted(self.image_dir.iterdir(), key=lambda name: float(name.stem.split("_")[-1]))
        image_labels = [label.split(maxsplit=3)[3:] for label in self.labels_file.read_text("utf-8").splitlines()]

        train_data, val_data = data_random_split(image_files, image_labels)
        if self.data_type == Types.train:
            image_files, image_labels = train_data
        elif self.data_type == Types.val:
            image_files, image_labels = val_data
        image_data = [(file, [{"bbox": None, "text": image_labels[index][0].replace("\u3000", " ")}])
                      for index, file in enumerate(image_files)]
        return image_data


class ICDAR2015Data:
    def __init__(self, data_type: Types.DataType, model_type: Types.ModelType = None) -> None:
        """
        ICDAR 2015 Dataset (det & rec) (en) (Word)
        source: https://rrc.cvc.uab.es/?ch=4
        """
        self.model_type, self.dataset_dir = model_type, DATASET_DIR / f"ICDAR 2015/{data_type}"
        self.image_dir, self.labels_dir = self.dataset_dir / "images", self.dataset_dir / "labels"
        self.ignore_tags = ["###"]

    def load_data(self) -> list:
        image_files, image_labels = list(self.image_dir.iterdir()), list(self.labels_dir.iterdir())
        image_data = []
        if self.model_type == Types.rec:
            for image_path, label_path in zip(image_files, image_labels):
                for label in label_path.read_text("utf-8-sig").splitlines():
                    label = label.split(',', maxsplit=8)
                    bbox, text = pairwise_tuples(tuple(map(float, label[:8]))), label[8:][0]
                    if text in self.ignore_tags:
                        continue
                    image_data.append((image_path, [{"bbox": bbox, "text": text}]))
        else:
            for image_path, label_path in zip(image_files, image_labels):
                image_bb_txt, labels = [], label_path.read_text("utf-8-sig").splitlines()
                for label in labels:
                    label = label.split(',', maxsplit=8)
                    bbox, text = pairwise_tuples(tuple(map(float, label[:8]))), label[8:][0]
                    if text in self.ignore_tags:
                        continue
                    image_bb_txt.append({"bbox": bbox, "text": text})
                if image_bb_txt:
                    image_data.append((image_path, image_bb_txt))
        return image_data


class ICDAR2017RCTWData:
    def __init__(self, data_type: Types.DataType, model_type: Types.ModelType = None) -> None:
        """
        ICDAR 2017 RCTW Dataset (det & rec) (ch) (Line)
        source: https://rctw.vlrlab.net/dataset
        """
        self.data_type, self.model_type, self.dataset_dir = data_type, model_type, DATASET_DIR / "ICDAR 2017 RCTW"
        self.image_dir, self.labels_dir = self.dataset_dir / "train_images", self.dataset_dir / "train_gts"
        self.ignore_tags = ["###"]

    def load_data(self) -> list:
        image_files, image_labels = list(self.image_dir.iterdir()), list(self.labels_dir.iterdir())
        train_data, val_data = data_random_split(image_files, image_labels)
        if self.data_type == Types.train:
            image_files, image_labels = train_data
        elif self.data_type == Types.val:
            image_files, image_labels = val_data
        image_data = []
        if self.model_type == Types.rec:
            for image_path, label_path in zip(image_files, image_labels):
                for label in label_path.read_text("utf-8-sig").splitlines():
                    label = label.split(',', maxsplit=9)
                    bbox, text = pairwise_tuples(tuple(map(float, label[:8]))), label[9:][0].strip('\"')
                    if text in self.ignore_tags:
                        continue
                    image_data.append((image_path, [{"bbox": bbox, "text": text}]))
        else:
            for image_path, label_path in zip(image_files, image_labels):
                image_bb_txt, labels = [], label_path.read_text("utf-8-sig").splitlines()
                for label in labels:
                    label = label.split(',', maxsplit=9)
                    bbox, text = pairwise_tuples(tuple(map(float, label[:8]))), label[9:][0].strip('\"')
                    if text in self.ignore_tags:
                        continue
                    image_bb_txt.append({"bbox": bbox, "text": text})
                if image_bb_txt:
                    image_data.append((image_path, image_bb_txt))
        return image_data


class ICDAR2019LSVTFullData:
    def __init__(self, data_type: Types.DataType, model_type: Types.ModelType = None) -> None:
        """
        ICDAR 2019 LSVT Dataset (det & rec) (en & ch) (Line)
        source: https://rrc.cvc.uab.es/?ch=16
        """
        self.data_type, self.model_type, self.dataset_dir = data_type, model_type, DATASET_DIR / "ICDAR 2019 LSVT"
        self.image_dir, self.labels_file = self.dataset_dir / "train_full", self.dataset_dir / "train_full_labels.json"
        self.ignore_tags = ["###"]

    def load_image_labels(self, image_files: list) -> list:
        with open(self.labels_file) as file:
            labels = json.load(file)
        image_data = []
        if self.model_type == Types.rec:
            for image_path in image_files:
                for label in labels[image_path.stem]:
                    bbox, text = label["points"], label["transcription"]
                    if text in self.ignore_tags:
                        continue
                    image_data.append((image_path, [{"bbox": bbox, "text": text}]))
        else:
            for image_path in image_files:
                image_bb_txt = []
                for label in labels[image_path.stem]:
                    bbox, text = label["points"], label["transcription"]
                    if text in self.ignore_tags:
                        continue
                    image_bb_txt.append({"bbox": bbox, "text": text})
                if image_bb_txt:
                    image_data.append((image_path, image_bb_txt))
        return image_data

    def load_data(self) -> list:
        image_files = list(self.image_dir.glob("*/*"))
        train_data, val_data = data_random_split(image_files)
        if self.data_type == Types.train:
            return self.load_image_labels(train_data)
        elif self.data_type == Types.val:
            return self.load_image_labels(val_data)


class ICDAR2019LSVTWeakData:
    def __init__(self, data_type: Types.DataType) -> None:
        """
        ICDAR 2019 LSVT Dataset (rec) (ch) (Line)
        source: https://rrc.cvc.uab.es/?ch=16
        """
        self.data_type, self.dataset_dir = data_type, DATASET_DIR / "ICDAR 2019 LSVT"
        self.image_dir, self.labels_file = self.dataset_dir / "train_weak", self.dataset_dir / "train_weak_labels.json"

    def load_image_labels(self, image_files: list) -> list:
        with open(self.labels_file, 'r') as file:
            labels = json.load(file)
        image_data = []
        for image_path in image_files:
            text = labels[image_path.stem][0]["transcription"]
            image_data.append((image_path, [{"bbox": None, "text": text}]))
        return image_data

    def load_data(self) -> list:
        image_files = list(self.image_dir.glob("*/*"))
        train_data, val_data = data_random_split(image_files)
        if self.data_type == Types.train:
            return self.load_image_labels(train_data)
        elif self.data_type == Types.val:
            return self.load_image_labels(val_data)


class MSRATD500:
    def __init__(self, data_type: Types.DataType) -> None:
        """
        2012/CVPR MSRA-TD500 Dataset (det) (en & ch) (Line)
        source: http://pages.ucsd.edu/%7Eztu/publication/MSRA-TD500.zip
        """
        self.dataset_dir = DATASET_DIR / f"MSRA-TD500/{data_type}"

    def load_data(self) -> list:
        image_files, image_data = list(self.dataset_dir.glob("*.jpg")), []
        for image_path in image_files:
            image_bb_txt, labels = [], image_path.with_suffix(".gt").read_text().splitlines()
            for label in labels:
                x, y, width, height, theta = map(float, label.split()[2:])
                bbox = rect_corners(x, y, width, height, theta)
                image_bb_txt.append({"bbox": bbox})
            if image_bb_txt:
                image_data.append((image_path, image_bb_txt))
        return image_data


class ICDAR2019ReCTS:

    def __init__(self, data_type: Types.DataType, model_type: Types.ModelType = None) -> None:
        """
        ICDAR 2019 ReCTS Dataset (det & rec) (en & ch) (Word)
        source: https://rrc.cvc.uab.es/?ch=12
        """
        self.data_type, self.model_type, self.dataset_dir = data_type, model_type, DATASET_DIR / "ICDAR 2019 ReCTS"
        self.image_dir, self.labels_dir = self.dataset_dir / "train/images", self.dataset_dir / "train/gt"
        self.ignore_tags = ["###"]

    def load_data(self) -> list:
        image_files, image_labels = list(self.image_dir.iterdir()), list(self.labels_dir.iterdir())
        train_data, val_data = data_random_split(image_files, image_labels)
        if self.data_type == Types.train:
            image_files, image_labels = train_data
        elif self.data_type == Types.val:
            image_files, image_labels = val_data
        image_data = []
        if self.model_type == Types.rec:
            for image_path, label_path in zip(image_files, image_labels):
                labels = json.loads(label_path.read_text("utf-8"))
                for label in labels["lines"]:
                    bbox, text = pairwise_tuples(label["points"]), label["transcription"]
                    if text in self.ignore_tags:
                        continue
                    image_data.append((image_path, [{"bbox": bbox, "text": text}]))
        else:
            for image_path, label_path in zip(image_files, image_labels):
                image_bb_txt, labels = [], json.loads(label_path.read_text("utf-8"))
                for label in labels["lines"]:
                    bbox, text = pairwise_tuples(label["points"]), label["transcription"]
                    if text in self.ignore_tags:
                        continue
                    image_bb_txt.append({"bbox": bbox, "text": text})
                if image_bb_txt:
                    image_data.append((image_path, image_bb_txt))
        return image_data


class SynthTextData:
    def __init__(self, data_type: Types.DataType, model_type: Types.ModelType = None) -> None:
        """
        SynthText Dataset (det & rec) (en) (Word & Line)
        source: https://github.com/ankush-me/SynthText
        """
        self.data_type, self.model_type, self.dataset_dir = data_type, model_type, DATASET_DIR / "SynthText"
        self.image_dir = self.dataset_dir / "images"
        self.labels_file, self.labels_json_file = self.dataset_dir / "gt.mat", self.dataset_dir / "gt.json"

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
            image_bb_txt = []
            for bbox, text in zip(bboxes, texts):
                image_bb_txt.append({"bbox": pairwise_tuples(bbox), "text": text})
            all_labels[name] = image_bb_txt
        with open(self.labels_json_file, "w") as outfile:
            json.dump(all_labels, outfile)

    def load_image_labels(self, image_files: list) -> list:
        if not self.labels_json_file.exists():
            self.convert_mat_to_json()

        with open(self.labels_json_file) as file:
            labels = json.load(file)

        image_data = []
        if self.model_type == Types.rec:
            for image_path in image_files:
                for label in labels[image_path.name]:
                    image_data.append((image_path, [label]))
        else:
            image_data = [(image_path, labels[image_path.name]) for image_path in image_files]
        return image_data

    def load_data(self) -> list:
        image_files = list(self.image_dir.glob("*/*"))
        train_data, val_data = data_random_split(image_files)
        if self.data_type == Types.train:
            return self.load_image_labels(train_data)
        elif self.data_type == Types.val:
            return self.load_image_labels(val_data)


class TextOCR01Data:
    def __init__(self, data_type: Types.DataType, model_type: Types.ModelType = None) -> None:
        """
        TextOCR v0.1 Dataset (det & rec) (en) (Word)
        source: https://textvqa.org/textocr/dataset/
        """
        self.model_type, self.dataset_dir = model_type, DATASET_DIR / "TextOCR V0.1"
        self.image_dir = self.dataset_dir / "train_val_images"
        self.labels_file = self.dataset_dir / f"TextOCR_0.1_{data_type}.json"

    def load_labels(self) -> dict:
        with open(self.labels_file) as file:
            labels = json.load(file)["anns"].values()

        all_labels = {}
        for val in labels:
            text = val["utf8_string"]
            if len(text) > 1:
                image_id = val["id"].split('_')[0]
                bbox = pairwise_tuples(val["points"])
                all_labels.setdefault(image_id, []).append({"bbox": bbox, "text": text})
        return all_labels

    def load_data(self) -> list:
        image_labels = self.load_labels()
        image_files = {file.stem: file for file in self.image_dir.iterdir()}
        image_data = []
        if self.model_type == Types.rec:
            for key, values in image_labels.items():
                for value in values:
                    image_data.append((image_files[key], [value]))
        else:
            image_data = [(image_files[key], value) for key, value in image_labels.items()]
        return image_data


class TRDGSyntheticData:
    def __init__(self, lang: Types.Language, data_type: Types.DataType) -> None:
        """
        Multilingual Dataset made using trdg package. (rec) (multi lang) (Word & Line)
        Language depends on what data was generated.
        source: https://github.com/Belval/TextRecognitionDataGenerator
        """
        self.data_type, self.dataset_dir = data_type, DATASET_DIR / "TRDG Synthetic Images"
        self.image_dir = self.dataset_dir / lang
        self.labels_file = self.image_dir / f"labels.txt"

    def load_image_labels(self, image_files: list) -> list:
        labels = {text.split(maxsplit=1)[0]: text.split(maxsplit=1)[1] for text in
                  self.labels_file.read_text("utf-8").splitlines()}
        image_data = [(image_file, [{"bbox": None, "text": labels[image_file.name]}]) for image_file in image_files]
        return image_data

    def load_data(self) -> list:
        image_files = list(self.image_dir.glob("*.jpg"))
        train_data, val_data = data_random_split(image_files)
        if self.data_type == Types.train:
            return self.load_image_labels(train_data)
        elif self.data_type == Types.val:
            return self.load_image_labels(val_data)


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


def merge_data_sources(*args) -> list:
    all_images_data = []
    for ds in args:
        image_data = ds.load_data()
        all_images_data.extend(image_data)
    return all_images_data


def load_data(lang: Types.Language, model_type: Types.ModelType, data_type: Types.DataType) -> list:
    if lang == Types.english:
        if model_type == Types.det:
            return merge_data_sources(
                ICDAR2015Data(data_type),
                MSRATD500(data_type),
                SynthTextData(data_type),
                TextOCR01Data(data_type)
            )
        elif model_type == Types.rec:
            return merge_data_sources(
                ICDAR2015Data(data_type, model_type),
                SynthTextData(data_type, model_type),
                TextOCR01Data(data_type, model_type),
                TRDGSyntheticData(lang, data_type)
            )
    elif lang == Types.chinese:
        if model_type == Types.det:
            return merge_data_sources(
                ICDAR2017RCTWData(data_type),
                ICDAR2019LSVTFullData(data_type),
                MSRATD500(data_type),
                ICDAR2019ReCTS(data_type),
            )
        elif model_type == Types.rec:
            return merge_data_sources(
                ChStreetViewTxtRecData(data_type),
                ICDAR2017RCTWData(data_type, model_type),
                ICDAR2019LSVTFullData(data_type, model_type),
                ICDAR2019LSVTWeakData(data_type),
                ICDAR2019ReCTS(data_type, model_type),
                TRDGSyntheticData(lang, data_type)
            )


if __name__ == '__main__':
    start = perf_counter()

    ts_data = load_data(Types.english, Types.det, Types.train)
    ts_len = len(ts_data)
    print(f"Data Source Length: {ts_len:,} Data Load Time: {perf_counter() - start:.4f}\n")
    for ts_idx in range(0, ts_len, round(ts_len / 200)):
        ts_image_path, ts_image_labels = ts_data[ts_idx]
        print(f"Image Path: {ts_image_path}\nImage Labels: {ts_image_labels}\n")
        visualize_data(str(ts_image_path), ts_image_labels)
