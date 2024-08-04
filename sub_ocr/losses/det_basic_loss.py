"""
This code is refer from:
https://github.com/WenmuZhou/DBNet.pytorch/blob/master/models/losses/basic_loss.py
"""

import torch
import torch.nn.functional as F
from torch import nn


class BalanceLoss(nn.Module):
    def __init__(self, balance_loss=True, main_loss_type="DiceLoss", negative_ratio=3, eps=1e-6):
        """
        The BalanceLoss for Differentiable Binarization text detection
        args:
            balance_loss (bool): whether balance loss or not, default is True
            main_loss_type (str): can only be one of ['CrossEntropy','DiceLoss','Euclidean','BCELoss', 'MaskL1Loss'],
                                default is  'DiceLoss'.
            negative_ratio (int|float): float, default is 3.
            eps (float): default is 1e-6.
        """
        super().__init__()
        self.balance_loss = balance_loss
        self.main_loss_type = main_loss_type
        self.negative_ratio = negative_ratio
        self.eps = eps

        if self.main_loss_type == "CrossEntropy":
            self.loss = nn.CrossEntropyLoss()
        elif self.main_loss_type == "Euclidean":
            self.loss = nn.MSELoss()
        elif self.main_loss_type == "DiceLoss":
            self.loss = DiceLoss(self.eps)
        elif self.main_loss_type == "BCELoss":
            self.loss = BCELoss(reduction="none")
        elif self.main_loss_type == "MaskL1Loss":
            self.loss = MaskL1Loss(self.eps)
        else:
            loss_type = ["CrossEntropy", "DiceLoss", "Euclidean", "BCELoss", "MaskL1Loss"]
            raise Exception(f"main_loss_type in BalanceLoss() can only be one of {loss_type}")

    def forward(self, pred, gt, mask=None):
        """
        The BalanceLoss for Differentiable Binarization text detection
        args:
            pred (variable): predicted feature maps.
            gt (variable): ground truth feature maps.
            mask (variable): masked maps.
        return: (variable) balanced loss
        """
        positive = gt * mask
        negative = (1 - gt) * mask

        positive_count = int(positive.sum())
        negative_count = int(min(negative.sum(), positive_count * self.negative_ratio))
        loss = self.loss(pred, gt, mask=mask)

        if not self.balance_loss:
            return loss

        positive_loss = positive * loss
        negative_loss = negative * loss

        negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)

        balance_loss = (positive_loss.sum() + negative_loss.sum()) / (positive_count + negative_count + self.eps)
        return balance_loss


class DiceLoss(nn.Module):
    """
    Dice Loss is to compare the similarity between the predicted text binary image and the label. It is often used in
    binary image segmentation.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, prediction: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, weights=None) -> float:
        """
        prediction: one or two heatmaps of shape (N, 1, H, W), the losses of tow heatmaps are added together.
        gt: (N, 1, H, W)
        mask: (N, H, W)
        """
        if prediction.dim() == 4:
            prediction = prediction[:, 0, :, :]
            gt = gt[:, 0, :, :]
        assert prediction.shape == gt.shape
        assert prediction.shape == mask.shape
        if weights is not None:
            assert weights.shape == mask.shape
            mask = weights * mask
        intersection = (prediction * gt * mask).sum()
        union = (prediction * mask).sum() + (gt * mask).sum() + self.eps
        loss = 1 - 2.0 * intersection / union
        assert loss <= 1
        return loss


class MaskL1Loss(nn.Module):
    """
    MaskL1 Loss is to calculate the 𝐿1 distance between the predicted text threshold map and the label.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, prediction: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor):
        loss = (torch.abs(prediction - gt) * mask).sum() / (mask.sum() + self.eps)
        # loss = torch.mean(loss)
        return loss


class BCELoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input_, label, **_):
        loss = F.binary_cross_entropy(input_, label, reduction=self.reduction)
        return loss
