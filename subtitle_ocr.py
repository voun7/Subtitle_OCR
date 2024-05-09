import numpy as np
import torch

from models.detection.db import DB, DBPostProcess
from models.recognition.crnn import CRNN
from utilities.utils import Types, read_image, resize_norm_img
from utilities.visualize import visualize_data


class SubtitleOCR:
    models_dir = r"C:\Users\Victor\OneDrive\Backups\Subtitle OCR Models"
    db_model_file = f"{models_dir}/DB deformable_resnet50 (0.448).pt"
    db_pp_model_file = r""
    crnn_model_file = r""

    def __init__(self) -> None:
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.det_model = self.init_model(Types.db)
        self.det_post_process = DBPostProcess()
        # self.rec_model = self.init_model(Types.crnn)
        # self.rec_post_process =

    def init_model(self, model: Types.ModelName):
        state = None
        if model is Types.db:
            model = DB({"name": Types.db, "backbone": "deformable_resnet50", "pretrained": True})
            state = torch.load(self.db_model_file)
        elif model is Types.crnn:
            model = CRNN({"": ""})
            state = torch.load(self.crnn_model_file)

        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        return model

    def preprocess_image(self, image: np.ndarray, pad: bool = False) -> torch.Tensor:
        image = resize_norm_img(image, 640, 640, pad)[0]
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image).to(self.device)
        return image

    @torch.no_grad()
    def ocr(self, image_path: str) -> list:
        image, image_height, image_width = read_image(image_path)
        image = self.preprocess_image(image)
        prediction = self.det_model(image)
        batch = {'shape': [(image_height, image_width)]}
        bboxes, scores = self.det_post_process(batch, prediction)
        labels = [{"bbox": bbox.tolist(), "text": None} for bbox, score in zip(bboxes[0], scores[0]) if score > 0.5]
        return labels


def test_main() -> None:
    test_sub_ocr = SubtitleOCR()
    test_image_file = r"C:\Users\Victor\Documents\Python Datasets\Subtitle_OCR\ICDAR 2015\train\images\img_56.jpg"
    test_output = test_sub_ocr.ocr(test_image_file)
    visualize_data(test_image_file, test_output)


if __name__ == '__main__':
    test_main()
