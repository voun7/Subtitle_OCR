import logging
import os
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import cv2 as cv
import numpy as np
import requests
import torch

from sub_ocr.modeling import build_model
from sub_ocr.postprocess import build_post_process
from sub_ocr.utils import read_image, normalize_img, pascal_voc_bb

torch.set_num_threads(1)  # improves cpu performance

logger = logging.getLogger(__name__)


class SubtitleOCR:
    supported_languages = ["en", "ch"]

    default_configs = {
        "det_en": "en_det_ppocr_v3",
        "det_ch": "ch_ptocr_v4_det_infer.pth",
        "rec_en": "en_PP-OCRv4_rec",
        "rec_ch": "ch_PP-OCRv4_rec",
    }

    configs = {
        "det": {
            "en": {
                "en_det_ppocr_v3": {
                    "Architecture": {'model_type': 'det', 'algorithm': 'DB', 'Transform': None,
                                     'Backbone': {'name': 'MobileNetV3', 'scale': 0.5, 'model_name': 'large',
                                                  'disable_se': True},
                                     'Neck': {'name': 'RSEFPN', 'out_channels': 96, 'shortcut': True},
                                     'Head': {'name': 'DBHead', 'k': 50}},
                    "params": {"height": 960, "width": 960, "m32": True, "sort_merge": True},
                    "PostProcess": {'name': 'DBPostProcess', 'thresh': 0.3, 'box_thresh': 0.6, 'max_candidates': 1000,
                                    'unclip_ratio': 2.5}
                },
            },
            "ch": {
                "ch_PP-OCRv4_det_student": {
                    "Architecture": {'model_type': 'det', 'algorithm': 'DB', 'Transform': None,
                                     'Backbone': {'name': 'PPLCNetV3', 'scale': 0.75, 'det': True},
                                     'Neck': {'name': 'RSEFPN', 'out_channels': 96, 'shortcut': True},
                                     'Head': {'name': 'DBHead', 'k': 50}},
                    "params": {"height": 640, "width": 640, "m32": True, "sort_merge": True},
                    "PostProcess": {'name': 'DBPostProcess', 'thresh': 0.3, 'box_thresh': 0.6, 'max_candidates': 1000,
                                    'unclip_ratio': 2.5}
                },
                "ch_ptocr_v4_det_infer.pth": {
                    "Architecture": {'model_type': 'det', 'algorithm': 'DB', 'Transform': None,
                                     'Backbone': {'name': 'PPLCNetV3', 'scale': 0.75, 'det': True},
                                     'Neck': {'name': 'RSEFPN', 'out_channels': 96, 'shortcut': True},
                                     'Head': {'name': 'DBHead', 'k': 50}},
                    "params": {"height": 640, "width": 960, "m32": True, "sort_merge": False},
                    "PostProcess": {'name': 'DBPostProcess', 'thresh': 0.3, 'box_thresh': 0.6, 'max_candidates': 1000,
                                    'unclip_ratio': 2.5}
                }
            },
        },
        "rec": {
            "en": {
                "en_PP-OCRv4_rec": {
                    "Architecture": {'model_type': 'rec', 'algorithm': 'SVTR_LCNet', 'Transform': None,
                                     'Backbone': {'name': 'PPLCNetV3', 'scale': 0.95},
                                     'Neck': {'name': 'SequenceEncoder', 'encoder_type': 'svtr', 'dims': 120,
                                              'depth': 2,
                                              'hidden_dims': 120, 'kernel_size': [1, 3], 'use_guide': True},
                                     'Head': {'name': 'CTCHead'}},
                    "params": {"height": 48, "width": 320},
                    "PostProcess": {'name': 'CTCLabelDecode'}
                },
            },
            "ch": {
                "ch_PP-OCRv4_rec": {
                    "Architecture": {'model_type': 'rec', 'algorithm': 'SVTR_LCNet', 'Transform': None,
                                     'Backbone': {'name': 'PPLCNetV3', 'scale': 0.95},
                                     'Neck': {'name': 'SequenceEncoder', 'encoder_type': 'svtr', 'dims': 120,
                                              'depth': 2,
                                              'hidden_dims': 120, 'kernel_size': [1, 3], 'use_guide': True},
                                     'Head': {'name': 'CTCHead'}},
                    "params": {"height": 48, "width": 320},
                    "PostProcess": {'name': 'CTCLabelDecode'}
                },
            },
        }
    }

    def __init__(self, lang: str, models_dir: str, device: str = "cuda") -> None:
        """
        Subtitle OCR package.
        :param lang: Language for text detection and recognition.
        :param models_dir: Directory for model files.
        :param device: Device to load model. GPU will only be used if it's requested and available.
        """
        assert lang in self.supported_languages, "Requested language is not available!"
        assert device in ["cpu", "cuda"], "Requested device is not available!"
        self.models_dir, self.device = Path(models_dir), device if torch.cuda.is_available() else "cpu"
        self.maybe_download_models()
        self.det_model, self.det_post_process, self.det_params = self.init_model(lang, "det")
        self.rec_model, self.rec_post_process, self.rec_params = self.init_model(lang, "rec")

    def maybe_download_models(self) -> None:
        """
        Download models from cloud if they are not available.
        """
        if not self.models_dir.exists() or len(list(self.models_dir.iterdir())) < 2:
            logger.warning("Models not found! Downloading models...")
            self.models_dir.mkdir(exist_ok=True)
            response = requests.get("https://www.dropbox.com/scl/fo/gkfzxqctfvnp600b9yy1x/"
                                    "ACIXdjd1JN2xjNX8ZKsuAHw?rlkey=zh2fzkz5gth8mohhb3gw2awe0&st=2jl1lq3e&dl=1")
            with ZipFile(BytesIO(response.content)) as zip_file:
                zip_file.extractall(self.models_dir)
                logger.warning(f"Models downloaded. Names: {zip_file.namelist()}")

    def init_model(self, lang: str, model_type: str) -> tuple:
        """
        Setup model and post processor.
        """
        config_name = self.default_configs[f"{model_type}_{lang}"]
        config = self.configs[model_type][lang][config_name]
        if ".pt" in config_name:
            model_file = self.models_dir / config_name
        else:
            model_file = next(self.models_dir.glob(f"{config_name} *.pt"))  # best loss will be used
        config.update({"lang": lang})
        model, post_processor = build_model(config), build_post_process(config)

        logger.debug(f"Device: {self.device}, Model Config: {config},\nModel File: {model_file}")
        model.load_state_dict(torch.load(model_file, self.device, weights_only=True))
        model.to(self.device).eval()
        return model, post_processor, config["params"]

    def det_image_resize(self, image: np.ndarray) -> np.ndarray:
        scale = min(self.det_params["height"] / image.shape[0], self.det_params["width"] / image.shape[1])
        resize_h, resize_w = image.shape[0] * scale, image.shape[1] * scale
        if self.det_params["m32"]:  # resized image shape as a multiple of 32.
            resize_h, resize_w = round(resize_h / 32) * 32, round(resize_w / 32) * 32
        # Resize the image while maintaining aspect ratio
        resized_image = cv.resize(image, (int(resize_w), int(resize_h)))
        return normalize_img(resized_image)

    def rec_image_resize(self, image: np.ndarray) -> np.ndarray:
        resize_h, resize_w = self.rec_params["height"], self.rec_params["width"]
        image_h, image_w = image.shape[:2]
        max_wh_ratio = max(resize_w / resize_h, image_w / image_h)
        resize_w, ratio = resize_h * max_wh_ratio, image_w / image_h
        if not resize_h * ratio > resize_w:
            resize_w = resize_h * ratio
        resized_image = cv.resize(image, (int(resize_w), resize_h))
        return normalize_img(resized_image)

    @staticmethod
    def merge_bboxes(bboxes: np.ndarray) -> tuple:
        n_x1, n_y1 = np.min(bboxes[:, 0], axis=0)
        n_x2, n_y2 = np.max(bboxes[:, 1], axis=0)
        n_x3, n_y3 = np.max(bboxes[:, 2], axis=0)
        n_x4, n_y4 = np.min(bboxes[:, 3], axis=0)
        return (n_x1, n_y1), (n_x2, n_y2), (n_x3, n_y3), (n_x4, n_y4)

    def sort_merge_bboxes(self, bboxes: np.ndarray, threshold: int = 10) -> list:
        """
        Sort and merge bboxes that are very close and on the same horizontal line to create larger bboxes.
        The y-coordinates is used because bounding boxes that are aligned horizontally will have similar y-coordinates.
        e.g, Single word bboxes that are close to each other would merge together to form a bbox containing a sentence.
        """
        # Calculate the average y-coordinate for each bbox
        avg_y = np.mean(bboxes[:, :, 1], axis=1)
        # Sort the bounding boxes by their average y-coordinate
        sorted_indices = np.argsort(avg_y)
        sorted_bboxes, sorted_avg_y = bboxes[sorted_indices], avg_y[sorted_indices]
        # Find the differences between consecutive average y-coordinates
        diff_y = np.diff(sorted_avg_y, prepend=sorted_avg_y[0])
        # Identify groups based on the threshold
        group_labels = np.cumsum(diff_y > threshold)
        # Use advanced indexing to group bounding boxes
        groups = [sorted_bboxes[group_labels == i] for i in np.unique(group_labels)]
        return [{"bbox": self.merge_bboxes(bbs)} for bbs in groups]

    def text_detector(self, image: np.ndarray, image_height: int, image_width: int) -> list | None:
        image = self.det_image_resize(image)
        image = torch.from_numpy(image).to(self.device)
        prediction = self.det_model(image.unsqueeze(0))
        batch = {"shape": [(image_height, image_width)]}
        bboxes, scores = self.det_post_process(batch, prediction)
        bboxes = bboxes[0][scores[0] > 0]  # Remove bbox indexes with a score of zero.
        if not bboxes.size:
            return
        return self.sort_merge_bboxes(bboxes) if self.det_params["sort_merge"] else [{"bbox": bb.tolist()} for bb in
                                                                                     bboxes]

    def recognizer(self, image: np.ndarray) -> tuple:
        image = self.rec_image_resize(image)
        image = torch.from_numpy(image).to(self.device)
        prediction = self.rec_model(image.unsqueeze(0))
        return self.rec_post_process(prediction)

    def text_recognizer(self, image: np.ndarray, labels: list | None) -> list:
        if labels is None:
            return []
        elif labels:  # for labels with bbox
            for label in labels:
                x_min, y_min, x_max, y_max = pascal_voc_bb(label["bbox"])
                cropped_image = image[y_min:y_max, x_min:x_max]  # crop image with bbox
                if cropped_image.size:
                    label["text"], label["score"] = self.recognizer(cropped_image)[0]
                else:
                    label["text"], label["score"] = "", 0  # for invalid crops
        else:
            labels = [{"text": text, "score": score} for text, score in self.recognizer(image)]
        return labels

    @torch.inference_mode()
    def ocr(self, image_path: str, det: bool = True, rec: bool = True) -> list:
        image, image_height, image_width = read_image(image_path)
        labels = self.text_detector(image, image_height, image_width) if det else []
        labels = self.text_recognizer(image, labels) if rec else labels
        return labels


def test_ocr() -> None:
    username = os.getlogin()
    test_image_files = Path(rf"C:\Users\{username}\OneDrive\Public\test images")
    test_sub_ocr = SubtitleOCR("ch", rf"C:\Users\{username}\OneDrive\Backups\Subtitle OCR Models")
    for test_image in test_image_files.iterdir():
        test_outputs = test_sub_ocr.ocr(str(test_image))
        logger.info(test_image)
        for test_output in test_outputs:
            logger.info(test_output)
        logger.info("")
        visualize_data(str(test_image), test_outputs, False, True)


if __name__ == '__main__':
    from utilities.logger_setup import setup_logging
    from utilities.visualize import visualize_data

    setup_logging()
    logger.debug("Logging Started")
    test_ocr()
    logger.debug("Logging Ended\n\n")
