"""
This code is refer from:
https://github.com/WenmuZhou/DBNet.pytorch/blob/master/models/losses/DB_loss.py
"""

import torch
from torch import nn

from .det_basic_loss import BalanceLoss, MaskL1Loss, DiceLoss


class DBLoss(nn.Module):
    """
     Differentiable Binarization (DB) Loss Function.
     Since three inference maps are obtained in the training, in the loss function, it is also necessary to combine
     these three maps and their real labels to build three parts of the loss function respectively.
    """

    def __init__(self, balance_loss=True, main_loss_type="DiceLoss", alpha=5, beta=10, ohem_ratio=3, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.dice_loss = DiceLoss(eps)
        self.l1_loss = MaskL1Loss(eps)
        self.bce_loss = BalanceLoss(balance_loss=balance_loss, main_loss_type=main_loss_type, negative_ratio=ohem_ratio)

    def forward(self, predictions: torch.Tensor, batch: dict) -> dict:
        """
        :params: predictions (train mode): prob map, thresh map, binary map
        :params: gts (eval mode): prob map, thresh map
        """
        assert predictions.dim() == 4
        shrink_maps = predictions[:, 0, :, :]
        threshold_maps = predictions[:, 1, :, :]
        binary_maps = None
        if predictions.size(1) == 3:  # size = 3 when model in train mode & 2 in eval mode
            binary_maps = predictions[:, 2, :, :]

        # 1. For the text inference probability map, use the binary cross-entropy loss function
        loss_shrink_maps = self.bce_loss(shrink_maps, batch['shrink_map'], batch['shrink_mask'])
        # 2. For the text inference threshold map, use the L1 distance loss function
        loss_threshold_maps = self.l1_loss(threshold_maps, batch['threshold_map'], batch['threshold_mask'])
        losses = dict(loss_shrink_maps=loss_shrink_maps, loss_threshold_maps=loss_threshold_maps)
        if predictions.size(1) == 3:
            # 3. For text inference binary graph, use the dice loss function
            loss_binary_maps = self.dice_loss(binary_maps, batch['shrink_map'], batch['shrink_mask'])
            losses['loss_binary_maps'] = loss_binary_maps
            # 4. Multiply different loss functions by different weights
            loss_all = self.alpha * loss_shrink_maps + self.beta * loss_threshold_maps + loss_binary_maps
            losses['loss'] = loss_all
        else:
            losses['loss'] = loss_shrink_maps
        return losses
