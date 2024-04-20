import torch
import torch.nn as nn


class OHEMBalanceCrossEntropyLoss(nn.Module):
    """
    Dice Loss (OHEM) uses Dice Loss with OHEM to improve the imbalance of positive and negative samples. OHEM is a
    special automatic sampling method that can automatically select difficult samples for loss calculation, thereby
    improving the training effect of the model.
    """

    def __init__(self, negative_ratio: int = 3, eps: float = 1e-6, reduction: str = 'none') -> None:
        super().__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps
        self.reduction = reduction

    def forward(self, prediction: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> float:
        """
        Args:
            prediction: shape :math:`(N, 1, H, W)`, the prediction of network
            gt: shape :math:`(N, 1, H, W)`, the target
            mask: shape :math:`(N, H, W)`, the mask indicates positive regions
        """
        positive = (gt * mask)
        negative = (1 - gt) * mask

        positive_count = int(positive.sum())
        no_negative_expect = int(positive_count * self.negative_ratio)
        no_negative_current = int(negative.sum())
        negative_count = min(no_negative_expect, no_negative_current)

        loss = nn.functional.binary_cross_entropy(prediction, gt, reduction=self.reduction)
        positive_loss = loss * positive
        negative_loss = loss * negative

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

    def forward(self, prediction: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> float:
        loss = (torch.abs(prediction - gt) * mask).sum() / (mask.sum() + self.eps)
        return loss


class DBLoss(nn.Module):
    """
     Differentiable Binarization (DB) Loss Function.
     Since three inference maps are obtained in the training, in the loss function, it is also necessary to combine
     these three maps and their real labels to build three parts of the loss function respectively.
    """

    def __init__(self, alpha: float = 1.0, beta: int = 10, ohem_ratio: int = 3, eps: float = 1e-6) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        # Declare different loss functions
        self.dice_loss = DiceLoss(eps)
        self.l1_loss = MaskL1Loss(eps)
        self.bce_loss = OHEMBalanceCrossEntropyLoss(ohem_ratio, eps)

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
