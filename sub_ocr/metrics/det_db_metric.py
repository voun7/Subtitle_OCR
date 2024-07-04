import numpy as np
import torch

from .eval_det_iou import DetectionIoUEvaluator
from ..postprocess.db_postprocess import DBPostProcess


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        return self


class QuadMetrics:
    def __init__(self):
        self.evaluator = DetectionIoUEvaluator()

    def measure(self, batch, output, box_thresh=0.6):
        """
        batch: image, polygons
        batch: a dict produced by dataloaders.
            image: tensor of shape (N, C, H, W).
            polygons: tensor of shape (N, K, 4, 2), the polygons of objective regions.
            shape: the original shape of images.
            filename: the original filenames of images.
        output: (polygons, ...)
        """
        results = []
        bbox_batch = batch['bboxes']
        prediction_bbox_batch = np.array(output[0])
        prediction_scores_batch = np.array(output[1])
        for polygons, prediction_polygons, prediction_scores in zip(bbox_batch, prediction_bbox_batch,
                                                                    prediction_scores_batch):
            gt = [dict(bbox=np.int64(polygons[i])) for i in range(len(polygons))]
            pred = []
            for i in range(prediction_polygons.shape[0]):
                if prediction_scores[i] >= box_thresh:
                    pred.append(dict(bbox=prediction_polygons[i, :, :].astype(np.int32)))
            results.append(self.evaluator.evaluate_image(gt, pred))
        return results

    def validate_measure(self, batch, output, box_thresh=0.6):
        return self.measure(batch, output, box_thresh)

    def evaluate_measure(self, batch, output):
        return self.measure(batch, output), np.linspace(0, batch['image'].shape[0]).tolist()

    def gather_measure(self, raw_metrics):
        raw_metrics = [image_metrics for batch_metrics in raw_metrics for image_metrics in batch_metrics]

        result = self.evaluator.combine_results(raw_metrics)

        precision = AverageMeter()
        recall = AverageMeter()
        f_measure = AverageMeter()

        precision.update(result['precision'], n=len(raw_metrics))
        recall.update(result['recall'], n=len(raw_metrics))
        f_measure_score = 2 * precision.val * recall.val / (precision.val + recall.val + 1e-8)
        f_measure.update(f_measure_score)
        return {'precision': precision, 'recall': recall, 'f_measure': f_measure}


class RunningScore:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)

        if np.sum((label_pred[mask] < 0)) > 0:
            print(label_pred[label_pred < 0])
        hist = np.bincount(n_class * label_true[mask].astype(int) +
                           label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            try:
                self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)
            except Exception as e:
                print(e)
                pass

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / (hist.sum() + 0.0001)
        acc_cls = np.diag(hist) / (hist.sum(axis=1) + 0.0001)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 0.0001)
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / (hist.sum() + 0.0001)
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))
        return {'Overall Acc': acc, 'Mean Acc': acc_cls, 'FreqW Acc': fwavacc, 'Mean IoU': mean_iu}, cls_iu

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


def cal_text_score(texts, gt_texts, training_masks, running_metric_text, thresh=0.5):
    """
    :param texts: preb_prob_map
    :param gt_texts: gt_prob_map
    :param training_masks: supervision map
    :param running_metric_text:
    :param thresh:
    """
    training_masks = training_masks.data.cpu().numpy()
    pred_text = texts.data.cpu().numpy() * training_masks
    pred_text[pred_text <= thresh] = 0
    pred_text[pred_text > thresh] = 1
    pred_text = pred_text.astype(np.int32)
    gt_text = gt_texts.data.cpu().numpy() * training_masks
    gt_text = gt_text.astype(np.int32)
    running_metric_text.update(gt_text, pred_text)
    score_text, _ = running_metric_text.get_scores()
    return score_text


class DBMetric:
    def __init__(self, no_classes: int = 2) -> None:
        self.post_process = DBPostProcess()
        self.quad_metrics = QuadMetrics()
        self.running_metric_text = RunningScore(no_classes)
        self.raw_metrics = []

    def __call__(self, predictions: torch.Tensor, batch: dict, validation: bool) -> dict:
        shrink_maps = predictions[:, 0, :, :]
        score_shrink_map = cal_text_score(shrink_maps, batch['shrink_map'], batch['shrink_mask'],
                                          self.running_metric_text, thresh=0.25)
        accuracy = score_shrink_map['Mean Acc']
        iou_shrink_map = score_shrink_map['Mean IoU']
        if validation:
            assert predictions.size(1) == 2 and predictions.size(0) == 1, "Validation batch size must be 1!"
            bboxes, scores = self.post_process(batch, predictions)
            raw_metric = self.quad_metrics.measure(batch, (bboxes, scores))
            self.raw_metrics.append(raw_metric)
        return dict(accuracy=accuracy, iou_shrink_map=iou_shrink_map)

    def gather_val_metrics(self) -> dict:
        """
        This method is validation metrics that are generated per epoch instead of per batch.
        """
        metrics = self.quad_metrics.gather_measure(self.raw_metrics)
        self.raw_metrics = []  # clear out previous raw metrics after gathering
        metrics["recall"] = metrics['recall'].avg
        metrics["precision"] = metrics['precision'].avg
        metrics["f_measure"] = metrics['f_measure'].avg
        return metrics
