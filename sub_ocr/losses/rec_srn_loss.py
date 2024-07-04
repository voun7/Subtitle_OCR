import torch
from torch import nn


class SRNLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_func = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, predicts, batch):
        predict = predicts["predict"]
        word_predict = predicts["word_out"]
        gsrm_predict = predicts["gsrm_out"]
        label = batch[1]

        casted_label = torch.cast(x=label, dtype="int64")
        casted_label = torch.reshape(casted_label, shape=[-1, 1])

        cost_word = self.loss_func(word_predict, label=casted_label)
        cost_gsrm = self.loss_func(gsrm_predict, label=casted_label)
        cost_vsfd = self.loss_func(predict, label=casted_label)

        cost_word = torch.reshape(torch.sum(cost_word), shape=[1])
        cost_gsrm = torch.reshape(torch.sum(cost_gsrm), shape=[1])
        cost_vsfd = torch.reshape(torch.sum(cost_vsfd), shape=[1])

        sum_cost = cost_word * 3.0 + cost_vsfd + cost_gsrm * 0.15
        return {"loss": sum_cost, "word_loss": cost_word, "img_loss": cost_vsfd}
