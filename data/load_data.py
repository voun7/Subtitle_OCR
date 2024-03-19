import json
from pathlib import Path

DATASET_DIR = r"C:\Users\Victor\Documents\Python Datasets\Subtitle_OCR"


# Text Detection Format Annotation
# --Image file path--       --Image annotation information--
# train_images/img_1.jpg    {"bboxes": [[310, 104, 416, 141], [310, 104, 416, 141]}]

# Text Recognition Format Annotation
# --Image file path--       --Image text--
# train_images/img_1.jpg    Sample Text


class TRDGSyntheticData:
    def __init__(self, lang: str, mode: str = "train") -> None:
        """
        Multilingual Dataset made using trdg package. Language depends on what data was generated.
        """
        self.dataset_dir = f"{DATASET_DIR}/TRDG Synthetic Images/{lang}"
        self.mode = mode

    def get_images(self) -> list:
        pass

    def get_labels(self) -> dict:
        pass

    def load_training_data(self) -> tuple:
        pass

    def load_validation_data(self) -> tuple:
        pass

    def load_data(self) -> tuple:
        if self.mode == "train":
            return self.load_training_data()
        elif self.mode == "val":
            return self.load_validation_data()


class TextOCR01Data:
    def __init__(self, mode: str = "train") -> None:
        """
        English Dataset
        source:
        """
        self.dataset_dir = f"{DATASET_DIR}/"
        self.mode = mode
        self.img_dir = Path(f"{self.dataset_dir}/train & val")
        self.labels_file = f"{self.dataset_dir}/TextOCR_0.1_{mode}.json"

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

    def load_training_data(self):
        pass

    def load_validation_data(self):
        pass

    def load_data(self) -> tuple:
        labels = self.get_labels()
        images = self.get_images(labels)
        return images, labels


class SynthTextData:
    def __init__(self, mode: str = "train") -> None:
        """
        English Dataset
        source:
        """
        self.dataset_dir = f"{DATASET_DIR}/"
        self.mode = mode

    def get_images(self) -> list:
        pass

    def get_labels(self) -> dict:
        pass

    def load_training_data(self) -> tuple:
        pass

    def load_validation_data(self) -> tuple:
        pass

    def load_data(self) -> tuple:
        if self.mode == "train":
            return self.load_training_data()
        elif self.mode == "val":
            return self.load_validation_data()


class ICDAR2019LSVTData:
    def __init__(self, mode: str = "train") -> None:
        """
        Chinese Dataset
        source:
        """
        self.dataset_dir = f"{DATASET_DIR}/"
        self.mode = mode

    def get_images(self) -> list:
        pass

    def get_labels(self) -> dict:
        pass

    def load_training_data(self) -> tuple:
        pass

    def load_validation_data(self) -> tuple:
        pass

    def load_data(self) -> tuple:
        if self.mode == "train":
            return self.load_training_data()
        elif self.mode == "val":
            return self.load_validation_data()


class ChStreetViewTxtRecData:
    def __init__(self, mode: str = "train") -> None:
        """
        Chinese Dataset (Recognition only)
        source:
        """
        self.dataset_dir = f"{DATASET_DIR}/"
        self.mode = mode

    def get_images(self) -> list:
        pass

    def get_labels(self) -> dict:
        pass

    def load_training_data(self) -> tuple:
        pass

    def load_validation_data(self) -> tuple:
        pass

    def load_data(self) -> tuple:
        if self.mode == "train":
            return self.load_training_data()
        elif self.mode == "val":
            return self.load_validation_data()


def merge_data_sources():
    pass


def load_data(lang: str, model_type: str, mode: str) -> tuple:
    if lang == "en":
        if model_type == "det":
            ds = TextOCR01Data(mode)
            return ds.load_data()
        elif model_type == "rec":
            ds = TRDGSyntheticData(lang, mode)
            return ds.load_data()
    elif lang == "ch":
        if model_type == "det":
            ds = ICDAR2019LSVTData(mode)
            return ds.load_data()
        elif model_type == "rec":
            ds = ChStreetViewTxtRecData(mode)
            return ds.load_data()


res = load_data("en", "det", "train")
print(res)
