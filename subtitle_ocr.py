import torch

from models.detection.db import DB, DBPostProcess
from models.recognition.crnn import CRNN, CRNNPostProcess
from utilities.utils import Types, read_image, resize_norm_img, pascal_voc_bb, flatten_iter
from utilities.visualize import visualize_data


class SubtitleOCR:

    def __init__(self, lang: Types.Language = Types.english) -> None:
        self.models_dir = r"C:\Users\Victor\OneDrive\Backups\Subtitle OCR Models"
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
            model = DB({"name": Types.db, "backbone": "deformable_resnet50", "pretrained": True})
            state = torch.load(f"{self.models_dir}/{lang} DB deformable_resnet50.pt")
            post_processor = DBPostProcess(box_thresh=0.5)
        else:
            with open(f"models/recognition/alphabets/{lang}.txt", encoding="utf-8") as file:
                alphabet = "".join([line.rstrip("\n") for line in file])
            model = CRNN(**{"image_height": 32, "channel_size": 3, "num_class": len(alphabet) + 1})
            state = torch.load(f"{self.models_dir}/{lang} CRNN ctc.pt")
            post_processor = CRNNPostProcess(alphabet)

        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
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
        texts, _ = self.rec_post_process(predictions)
        for idx, labels in enumerate(image_labels):
            labels["text"] = texts[idx]
        return image_labels

    @torch.no_grad()
    def ocr(self, image_path: str) -> list:
        image, labels = self.text_detector(image_path)
        labels = self.text_recognizer(image, labels)
        return labels


def test_main() -> None:
    test_sub_ocr = SubtitleOCR()
    test_image_file = r"C:\Users\Victor\Documents\Python Datasets\Subtitle_OCR\ICDAR 2015\train\images\img_10.jpg"
    test_outputs = test_sub_ocr.ocr(test_image_file)
    for output in test_outputs:
        print(output)
    visualize_data(test_image_file, test_outputs, False, True)


if __name__ == '__main__':
    test_main()
