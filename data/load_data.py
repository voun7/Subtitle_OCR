import json
from pathlib import Path

import numpy as np

DATASET_DIR = r"C:\Users\Victor\Documents\Python Datasets\Subtitle_OCR"


# Text Detection Format Annotation
# --Image file path--       --Image annotation information--
# train_images/img_1.jpg    {"bboxes": [[310, 104, 416, 141], [310, 104, 416, 141]}]

# Text Recognition Format Annotation
# --Image file path--       --Image text--
# train_images/img_1.jpg    Sample Text


class TRDGSyntheticData:
    def __init__(self, lang: str, data_type: str = "train") -> None:
        """
        Multilingual Dataset made using trdg package. (rec) (multi lang)
        Language depends on what data was generated.
        source: https://github.com/Belval/TextRecognitionDataGenerator
        """
        self.dataset_dir = f"{DATASET_DIR}/TRDG Synthetic Images/{lang}"
        self.data_type = data_type

    def get_images(self) -> list:
        pass

    def get_labels(self) -> dict:
        pass

    def load_data(self) -> tuple:
        labels = self.get_labels()
        images = self.get_images()
        return images, labels


class TextOCR01Data:
    def __init__(self, data_type: str = "train") -> None:
        """
        TextOCR v0.1 Dataset (det) (en)
        source: https://textvqa.org/textocr/dataset/
        """
        self.dataset_dir = f"{DATASET_DIR}/TextOCR V0.1"
        self.img_dir = Path(f"{self.dataset_dir}/{data_type}")
        self.labels_file = f"{self.dataset_dir}/TextOCR_0.1_{data_type}.json"

    def get_images(self, labels: dict) -> list:
        """
        Images not in the label will be removed from image list.
        """
        return [img for img in self.img_dir.iterdir() if img.stem in set(labels.keys())]

    def get_labels(self) -> dict:
        with open(self.labels_file) as f:
            labels = f.read()
        labels = json.loads(labels)
        anns = labels["anns"]

        labels = {}
        for val in anns.values():
            img_id = val["id"].split('_')[0]
            bbox = val["bbox"]
            labels.setdefault(img_id, []).append(bbox)
        return labels

    def load_data(self) -> tuple:
        labels = self.get_labels()
        images = self.get_images(labels)
        return images, labels


class SynthTextData:
    def __init__(self, data_type: str = "train") -> None:
        """
        SynthText Dataset (det & rec) (en)
        source: https://github.com/ankush-me/SynthText
        """
        self.dataset_dir = f"{DATASET_DIR}/SynthText"
        self.data_type = data_type

    def get_images(self) -> list:
        pass

    def get_labels(self) -> dict:
        pass

    def load_data(self) -> tuple:
        labels = self.get_labels()
        images = self.get_images()
        return images, labels


class ICDAR2017RCTWData:
    def __init__(self, data_type: str = "train") -> None:
        """
        ICDAR 2017 RCTW Dataset (det & rec) (en & ch)
        source: https://rctw.vlrlab.net/dataset
        """
        self.dataset_dir = f"{DATASET_DIR}/ICDAR2017 RCTW"
        self.data_type = data_type

    def get_images(self) -> list:
        pass

    def get_labels(self) -> dict:
        pass

    def load_data(self) -> tuple:
        labels = self.get_labels()
        images = self.get_images()
        return images, labels


class ICDAR2019LSVTData:
    def __init__(self, data_type: str = "train") -> None:
        """
        ICDAR 2019 LSVT Dataset (det & rec) (en & ch)
        source: https://rrc.cvc.uab.es/?ch=16
        """
        self.dataset_dir = f"{DATASET_DIR}/ICDAR2019 LSVT"
        self.data_type = data_type

    def get_images(self) -> list:
        pass

    def get_labels(self) -> dict:
        pass

    def load_data(self) -> tuple:
        labels = self.get_labels()
        images = self.get_images()
        return images, labels


class COCOText2014Data:
    def __init__(self, data_type: str = "train") -> None:
        """
        COCO-Text v2.0 2017 Dataset (det & rec) (en)
        source: https://bgshih.github.io/cocotext/
        """
        self.dataset_dir = f"{DATASET_DIR}/COCOText V2"
        self.data_type = data_type

    def get_images(self) -> list:
        pass

    def get_labels(self) -> dict:
        pass

    def load_data(self) -> tuple:
        labels = self.get_labels()
        images = self.get_images()
        return images, labels


class COCOText2017Data:
    def __init__(self, data_type: str = "train") -> None:
        """
        COCO-Text v2.0 2017 Dataset (det & rec) (en)
        source: https://bgshih.github.io/cocotext/
        """
        self.dataset_dir = f"{DATASET_DIR}/COCOText V2"
        self.data_type = data_type

    def get_images(self) -> list:
        pass

    def get_labels(self) -> dict:
        pass

    def load_data(self) -> tuple:
        labels = self.get_labels()
        images = self.get_images()
        return images, labels


class ChStreetViewTxtRecData:
    def __init__(self, data_type: str = "train") -> None:
        """
        Chinese Scene Text Recognition Dataset (rec) (ch)
        source: https://aistudio.baidu.com/competition/detail/8/0/related-material
        """
        self.dataset_dir = f"{DATASET_DIR}/Chinese Street View Text Recognition"
        self.data_type = data_type

    def get_images(self) -> list:
        pass

    def get_labels(self) -> dict:
        pass

    def load_data(self) -> tuple:
        labels = self.get_labels()
        images = self.get_images()
        return images, labels


def data_random_split(files: object, split: float = 0.8, seed: int = 31) -> tuple[np.array, np.array]:
    """
    Randomly split any given files into two sets of non-overlapping training and validation files.
    :param files: Index able object with length.
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
    return train_files, val_files


def merge_data_sources(*args) -> tuple:
    all_images, all_labels = [], {}
    for dt in args:
        images, labels = dt.load_data()
        assert len(images) == len(labels)
        all_images.append(images)
        all_labels.update(labels)
    return all_images, all_labels


def load_data(lang: str, model_type: str, data_type: str) -> tuple:
    if lang == "en":
        if model_type == "det":
            ds1 = TextOCR01Data(data_type)
            # ds2 = SynthTextData(data_type)
            # ds3 = COCOTextV2Data(data_type)
            return ds1.load_data()
            # return merge_data_sources(ds1, ds2, ds3)
        elif model_type == "rec":
            ds1 = TRDGSyntheticData(lang, data_type)
            ds2 = SynthTextData(data_type)
            ds3 = COCOText2014Data(data_type)
            return merge_data_sources(ds1, ds2, ds3)
    elif lang == "ch":
        if model_type == "det":
            ds1 = ICDAR2019LSVTData(data_type)
            ds2 = ICDAR2017RCTWData(data_type)
            return merge_data_sources(ds1, ds2)
        elif model_type == "rec":
            ds1 = ICDAR2019LSVTData(data_type)
            ds2 = ICDAR2017RCTWData(data_type)
            ds3 = ChStreetViewTxtRecData(data_type)
            ds4 = TRDGSyntheticData(lang, data_type)
            return merge_data_sources(ds1, ds2, ds3, ds4)


res = load_data("en", "det", "train")
print(res)
