"""
This code is refer from:
https://github.com/whai362/PSENet/blob/python3/models/head/psenet_head.py
https://github.com/whai362/PSENet/blob/python3/models/loss/iou.py
"""

import torch
from torch import nn
from torch.nn import functional as F


def iou_single(a, b, mask, n_class):
    EPS = 1e-6
    valid = mask == 1
    a = a.masked_select(valid)
    b = b.masked_select(valid)
    miou = []
    for i in range(n_class):
        if a.shape == [0] and a.shape == b.shape:
            inter = torch.tensor(0.0)
            union = torch.tensor(0.0)
        else:
            inter = ((a == i).logical_and(b == i)).type(torch.float32)
            union = ((a == i).logical_or(b == i)).type(torch.float32)
        miou.append(torch.sum(inter) / (torch.sum(union) + EPS))
    miou = sum(miou) / len(miou)
    return miou


def iou(a, b, mask, n_class=2, reduce=True):
    batch_size = a.shape[0]

    a = a.reshape([batch_size, -1])
    b = b.reshape([batch_size, -1])
    mask = mask.reshape([batch_size, -1])

    iou = torch.zeros((batch_size,), dtype=torch.float32)
    for i in range(batch_size):
        iou[i] = iou_single(a[i], b[i], mask[i], n_class)

    if reduce:
        iou = torch.mean(iou)
    return iou


class PSELoss(nn.Module):
    def __init__(self, alpha, ohem_ratio=3, kernel_sample_mask="pred", reduction="sum", eps=1e-6, **kwargs):
        """Implement PSE Loss."""
        super().__init__()
        assert reduction in ["sum", "mean", "none"]
        self.alpha = alpha
        self.ohem_ratio = ohem_ratio
        self.kernel_sample_mask = kernel_sample_mask
        self.reduction = reduction
        self.eps = eps

    def forward(self, outputs, labels):
        predicts = outputs["maps"]
        predicts = F.interpolate(predicts, scale_factor=4)

        texts = predicts[:, 0, :, :]
        kernels = predicts[:, 1:, :, :]
        gt_texts, gt_kernels, training_masks = labels[1:]

        # text loss
        selected_masks = self.ohem_batch(texts, gt_texts, training_masks)

        loss_text = self.dice_loss(texts, gt_texts, selected_masks)
        iou_text = iou((texts > 0).type("int64"), gt_texts, training_masks, reduce=False)
        losses = dict(loss_text=loss_text, iou_text=iou_text)

        # kernel loss
        loss_kernels = []
        if self.kernel_sample_mask == "gt":
            selected_masks = gt_texts * training_masks
        elif self.kernel_sample_mask == "pred":
            selected_masks = (F.sigmoid(texts) > 0.5).type(torch.float32) * training_masks

        for i in range(kernels.shape[1]):
            kernel_i = kernels[:, i, :, :]
            gt_kernel_i = gt_kernels[:, i, :, :]
            loss_kernel_i = self.dice_loss(kernel_i, gt_kernel_i, selected_masks)
            loss_kernels.append(loss_kernel_i)
        loss_kernels = torch.mean(torch.stack(loss_kernels, dim=1), dim=1)
        iou_kernel = iou(
            (kernels[:, -1, :, :] > 0).type("int64"),
            gt_kernels[:, -1, :, :],
            training_masks * gt_texts,
            reduce=False,
        )
        losses.update(dict(loss_kernels=loss_kernels, iou_kernel=iou_kernel))
        loss = self.alpha * loss_text + (1 - self.alpha) * loss_kernels
        losses["loss"] = loss
        if self.reduction == "sum":
            losses = {x: torch.sum(v) for x, v in losses.items()}
        elif self.reduction == "mean":
            losses = {x: torch.mean(v) for x, v in losses.items()}
        return losses

    def dice_loss(self, input, target, mask):
        input = F.sigmoid(input)

        input = input.reshape([input.shape[0], -1])
        target = target.reshape([target.shape[0], -1])
        mask = mask.reshape([mask.shape[0], -1])

        input = input * mask
        target = target * mask

        a = torch.sum(input * target, 1)
        b = torch.sum(input * input, 1) + self.eps
        c = torch.sum(target * target, 1) + self.eps
        d = (2 * a) / (b + c)
        return 1 - d

    def ohem_single(self, score, gt_text, training_mask, ohem_ratio=3):
        pos_num = int(torch.sum((gt_text > 0.5).type(torch.float32))) - int(
            torch.sum(torch.logical_and((gt_text > 0.5), (training_mask <= 0.5)).type(torch.float32)))

        if pos_num == 0:
            selected_mask = training_mask
            selected_mask = selected_mask.reshape([1, selected_mask.shape[0], selected_mask.shape[1]]).type(
                torch.float32)
            return selected_mask

        neg_num = int(torch.sum((gt_text <= 0.5).type(torch.float32)))
        neg_num = int(min(pos_num * ohem_ratio, neg_num))

        if neg_num == 0:
            selected_mask = training_mask
            selected_mask = selected_mask.reshape([1, selected_mask.shape[0], selected_mask.shape[1]]).type(
                torch.float32)
            return selected_mask

        neg_score = torch.masked_select(score, gt_text <= 0.5)
        neg_score_sorted = torch.sort(-neg_score)
        threshold = -neg_score_sorted[neg_num - 1]

        selected_mask = torch.logical_and(torch.logical_or(score >= threshold, (gt_text > 0.5)),
                                          (training_mask > 0.5))
        selected_mask = selected_mask.reshape([1, selected_mask.shape[0], selected_mask.shape[1]]).type(torch.float32)
        return selected_mask

    def ohem_batch(self, scores, gt_texts, training_masks, ohem_ratio=3):
        selected_masks = []
        for i in range(scores.shape[0]):
            selected_masks.append(
                self.ohem_single(
                    scores[i, :, :],
                    gt_texts[i, :, :],
                    training_masks[i, :, :],
                    ohem_ratio,
                )
            )

        selected_masks = torch.concat(selected_masks, 0).type(torch.float32)
        return selected_masks
