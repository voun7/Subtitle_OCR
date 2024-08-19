import torch
import torch.nn as nn


class CTCLoss(nn.Module):

    def __init__(self) -> None:
        """
        Calculate the loss using CTC Loss
        """
        super().__init__()
        self.loss_func = nn.CTCLoss(zero_infinity=True)

    def forward(self, predicts: torch.Tensor, batch: dict) -> dict:
        N, B, _ = predicts.shape
        preds_lengths = torch.tensor([N] * B)
        loss = self.loss_func(predicts, batch["label"], preds_lengths, batch["length"])
        return {"loss": loss}
