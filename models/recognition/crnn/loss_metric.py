import torch
import torch.nn as nn

from models.recognition.crnn.post_process import StrLabelConverter, CRNNPostProcess


class CRNNLoss(nn.Module):

    def __init__(self, alphabet: str) -> None:
        """
        Calculate the loss using CTC Loss
        """
        super().__init__()
        self.converter = StrLabelConverter(alphabet)
        self.loss_func = nn.CTCLoss(zero_infinity=True)

    def forward(self, predictions: torch.Tensor, batch: dict) -> dict:
        prediction_size = torch.LongTensor([predictions.size(0)] * predictions.size(1))
        text, text_lengths = self.converter.encode(batch["text"])
        loss = self.loss_func(predictions, text, prediction_size, text_lengths)
        return {"loss": loss}


class CRNNMetrics:
    def __init__(self, alphabet: str, ignore_space: bool = True) -> None:
        self.post_process = CRNNPostProcess(alphabet)
        self.ignore_space = ignore_space

    def __call__(self, predictions: torch.Tensor, batch: dict, validation: bool) -> dict:
        correct_num = all_num = 0
        predictions = self.post_process(predictions)[0]
        for prediction, text in zip(predictions, batch["text"]):
            if self.ignore_space:
                prediction, text = prediction.replace(" ", ""), text.replace(" ", "")
            if prediction == text:
                correct_num += 1
            all_num += 1
        correct_num += correct_num
        all_num += all_num
        if validation:
            pass
        return {"accuracy": correct_num / all_num}
