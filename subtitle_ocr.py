import logging
from pathlib import Path

import torch

from models.detection.db import DB, DBPostProcess
from models.recognition.crnn import CRNN, CRNNPostProcess
from utilities.logger_setup import setup_logging
from utilities.utils import Types, read_image, resize_norm_img, pascal_voc_bb, flatten_iter, read_chars
from utilities.visualize import visualize_data

logger = logging.getLogger(__name__)


class SubtitleOCR:

    def __init__(self, lang: Types.Language = Types.english) -> None:
        self.models_dir = Path(r"C:\Users\Victor\OneDrive\Backups\Subtitle OCR Models")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.det_h, self.det_w = 640, 640
        self.det_model, self.det_post_process = self.init_model(lang)
        self.rec_h, self.rec_w = 32, 160
        self.rec_model, self.rec_post_process = self.init_model(lang, Types.rec)

    def init_model(self, lang: Types.Language, model: Types.ModelType = Types.det) -> tuple:
        """
        Setup model and post processor.
        """
        if model is Types.det:
            model_params = {"name": Types.db, "backbone": "deformable_resnet50", "pretrained": True}
            model = DB(model_params)
            file = next(self.models_dir.glob(f"{lang} DB deformable_resnet50 *.pt"))  # get first matching file
            post_processor = DBPostProcess(box_thresh=0.5)
        else:
            alphabet = read_chars(lang)
            model_params = {"image_height": self.rec_h, "channel_size": 3, "num_class": len(alphabet) + 1}
            model = CRNN(**model_params)
            file = next(self.models_dir.glob(f"{lang} CRNN ctc *.pt"))
            post_processor = CRNNPostProcess(alphabet)

        logger.debug(f"Model Params: {model_params} File: {file}")
        model.load_state_dict(torch.load(file))
        model.to(self.device).eval()
        return model, post_processor

    def text_detector(self, image_path: str) -> tuple:
        image, image_height, image_width = read_image(image_path)
        tensor_image = resize_norm_img(image, self.det_h, self.det_w, False)[0]
        tensor_image = torch.from_numpy(tensor_image).to(self.device)
        prediction = self.det_model(tensor_image.unsqueeze(0))
        batch = {"shape": [(image_height, image_width)]}
        bboxes, scores = self.det_post_process(batch, prediction)
        return image, [{"bbox": bbox.tolist()} for bbox, score in zip(bboxes[0], scores[0]) if score]

    def text_recognizer(self, image, image_labels: list) -> list:
        rec_batch = []
        for batch in image_labels:
            x_min, y_min, x_max, y_max = map(int, pascal_voc_bb(tuple(flatten_iter(batch["bbox"]))))
            tensor_image = image[y_min:y_max, x_min:x_max]
            tensor_image = resize_norm_img(tensor_image, self.rec_h, self.rec_w)[0]
            tensor_image = torch.from_numpy(tensor_image).to(self.device)
            rec_batch.append(tensor_image)
        predictions = self.rec_model(torch.stack(rec_batch))
        texts, scores = self.rec_post_process(predictions)
        for idx, labels in enumerate(image_labels):
            labels["text"], labels["score"] = texts[idx], scores[idx]
        return image_labels

    @torch.no_grad()
    def ocr(self, image_path: str) -> list:
        image, labels = self.text_detector(image_path)
        labels = self.text_recognizer(image, labels)
        return labels


def test_ocr() -> None:
    test_sub_ocr = SubtitleOCR()
    test_image_file = r"C:\Users\Victor\OneDrive\Public\test img1.png"
    test_outputs = test_sub_ocr.ocr(test_image_file)
    for output in test_outputs:
        print(output)
    visualize_data(test_image_file, test_outputs, False, True)


if __name__ == '__main__':
    setup_logging()
    logger.debug("\n\nLogging Started")
    test_ocr()
    logger.debug("Logging Ended\n\n")
