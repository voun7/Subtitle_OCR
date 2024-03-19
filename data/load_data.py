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
        Multilingual Dataset made using trdg package. (rec) (multi lang)
        Language depends on what data was generated.
        source: https://github.com/Belval/TextRecognitionDataGenerator
        """
        self.dataset_dir = f"{DATASET_DIR}/TRDG Synthetic Images/{lang}"
        self.mode = mode

    def get_images(self) -> list:
        pass

    def get_labels(self) -> dict:
        pass

    def load_data(self) -> tuple:
        labels = self.get_labels()
        images = self.get_images()
        return images, labels


class TextOCR01Data:
    def __init__(self, mode: str = "train") -> None:
        """
        TextOCR v0.1 Dataset (det) (en)
        source: https://textvqa.org/textocr/dataset/
        """
        self.dataset_dir = f"{DATASET_DIR}/TextOCR V0.1"
        self.img_dir = Path(f"{self.dataset_dir}/{mode}")
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

    def load_data(self) -> tuple:
        labels = self.get_labels()
        images = self.get_images(labels)
        return images, labels


class SynthTextData:
    def __init__(self, mode: str = "train") -> None:
        """
        SynthText Dataset (det & rec) (en)
        source: https://github.com/ankush-me/SynthText
        """
        self.dataset_dir = f"{DATASET_DIR}/SynthText"
        self.mode = mode

    def get_images(self) -> list:
        pass

    def get_labels(self) -> dict:
        pass

    def load_data(self) -> tuple:
        labels = self.get_labels()
        images = self.get_images()
        return images, labels


class ICDAR2017RCTWData:
    def __init__(self, mode: str = "train") -> None:
        """
        ICDAR 2017 RCTW Dataset (det & rec) (en & ch)
        source: https://rctw.vlrlab.net/dataset
        """
        self.dataset_dir = f"{DATASET_DIR}/ICDAR2017 RCTW"
        self.mode = mode

    def get_images(self) -> list:
        pass

    def get_labels(self) -> dict:
        pass

    def load_data(self) -> tuple:
        labels = self.get_labels()
        images = self.get_images()
        return images, labels


class ICDAR2019LSVTData:
    def __init__(self, mode: str = "train") -> None:
        """
        ICDAR 2019 LSVT Dataset (det & rec) (en & ch)
        source: https://rrc.cvc.uab.es/?ch=16
        """
        self.dataset_dir = f"{DATASET_DIR}/ICDAR2019 LSVT"
        self.mode = mode

    def get_images(self) -> list:
        pass

    def get_labels(self) -> dict:
        pass

    def load_data(self) -> tuple:
        labels = self.get_labels()
        images = self.get_images()
        return images, labels


class COCOTextV2Data:
    def __init__(self, mode: str = "train") -> None:
        """
        COCO-Text v2.0 2017 Dataset (det & rec) (en)
        source: https://bgshih.github.io/cocotext/
        """
        self.dataset_dir = f"{DATASET_DIR}/COCOText V2"
        self.mode = mode

    def get_images(self) -> list:
        pass

    def get_labels(self) -> dict:
        pass

    def load_data(self) -> tuple:
        labels = self.get_labels()
        images = self.get_images()
        return images, labels


class ChStreetViewTxtRecData:
    def __init__(self, mode: str = "train") -> None:
        """
        Chinese Scene Text Recognition Dataset (rec) (ch)
        source: https://aistudio.baidu.com/competition/detail/8/0/related-material
        """
        self.dataset_dir = f"{DATASET_DIR}/Chinese Street View Text Recognition"
        self.mode = mode

    def get_images(self) -> list:
        pass

    def get_labels(self) -> dict:
        pass

    def load_data(self) -> tuple:
        labels = self.get_labels()
        images = self.get_images()
        return images, labels


def merge_data_sources(*args) -> tuple:
    all_images, all_labels = [], {}
    for dt in args:
        images, labels = dt.load_data()
        assert len(images) == len(labels)
        all_images.append(images)
        all_labels.update(labels)
    return all_images, all_labels


def load_data(lang: str, model_type: str, mode: str) -> tuple:
    if lang == "en":
        if model_type == "det":
            ds1 = TextOCR01Data(mode)
            # ds2 = SynthTextData(mode)
            # ds3 = COCOTextV2Data(mode)
            return ds1.load_data()
            # return merge_data_sources(ds1, ds2, ds3)
        elif model_type == "rec":
            ds1 = TRDGSyntheticData(lang, mode)
            ds2 = SynthTextData(mode)
            ds3 = COCOTextV2Data(mode)
            return merge_data_sources(ds1, ds2, ds3)
    elif lang == "ch":
        if model_type == "det":
            ds1 = ICDAR2019LSVTData(mode)
            ds2 = ICDAR2017RCTWData(mode)
            return merge_data_sources(ds1, ds2)
        elif model_type == "rec":
            ds1 = ICDAR2019LSVTData(mode)
            ds2 = ICDAR2017RCTWData(mode)
            ds3 = ChStreetViewTxtRecData(mode)
            ds4 = TRDGSyntheticData(lang, mode)
            return merge_data_sources(ds1, ds2, ds3, ds4)


res = load_data("en", "det", "train")
print(res)
