from .eval_det_iou import DetectionIoUEvaluator


class DetMetric:
    def __init__(self, main_indicator="hmean"):
        self.evaluator = DetectionIoUEvaluator()
        self.main_indicator = main_indicator
        self.reset()

    def __call__(self, preds, batch, **kwargs):
        """
        batch: a list produced by dataloaders.
            image: np.ndarray  of shape (N, C, H, W).
            ratio_list: np.ndarray  of shape(N,2)
            polygons: np.ndarray  of shape (N, K, 4, 2), the polygons of objective regions.
        preds: a list of dict produced by post process
             points: np.ndarray of shape (N, K, 4, 2), the polygons of objective regions.
        """
        gt_polyons_batch = batch[2]
        for pred, gt_polyons in zip(preds, gt_polyons_batch):
            # prepare gt
            gt_info_list = [{"bbox": gt_polyon, "text": ""} for gt_polyon in gt_polyons]
            # prepare det
            det_info_list = [{"bbox": det_polyon, "text": ""} for det_polyon in pred["bbox"]]
            result = self.evaluator.evaluate_image(gt_info_list, det_info_list)
            self.results.append(result)

    def get_metric(self):
        """
        return metrics { 'precision': 0, 'recall': 0, 'hmean': 0}
        """
        metrics = self.evaluator.combine_results(self.results)
        self.reset()
        return metrics

    def reset(self):
        self.results = []  # clear results
