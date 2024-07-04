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


class DetFCEMetric:
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
            ignore_tags: np.ndarray  of shape (N, K), indicates whether a region is ignorable or not.
        preds: a list of dict produced by post process
             points: np.ndarray of shape (N, K, 4, 2), the polygons of objective regions.
        """
        gt_polyons_batch = batch[2]

        for pred, gt_polyons in zip(preds, gt_polyons_batch):
            # prepare gt
            gt_info_list = [{"bbox": gt_polyon, "text": ""} for gt_polyon, ignore_tag in gt_polyons]
            # prepare det
            det_info_list = [{"bbox": det_polyon, "text": "", "score": score}
                             for det_polyon, score in zip(pred["bbox"], pred["scores"])]

            for score_thr in self.results.keys():
                det_info_list_thr = [det_info for det_info in det_info_list if det_info["score"] >= score_thr]
                result = self.evaluator.evaluate_image(gt_info_list, det_info_list_thr)
                self.results[score_thr].append(result)

    def get_metric(self):
        """
        return metrics {'heman':0,
            'thr 0.3':'precision: 0 recall: 0 hmean: 0',
            'thr 0.4':'precision: 0 recall: 0 hmean: 0',
            'thr 0.5':'precision: 0 recall: 0 hmean: 0',
            'thr 0.6':'precision: 0 recall: 0 hmean: 0',
            'thr 0.7':'precision: 0 recall: 0 hmean: 0',
            'thr 0.8':'precision: 0 recall: 0 hmean: 0',
            'thr 0.9':'precision: 0 recall: 0 hmean: 0',
            }
        """
        metrics = {}
        hmean = 0
        for score_thr in self.results.keys():
            metric = self.evaluator.combine_results(self.results[score_thr])
            metric_str = f"precision:{metric["precision"]:.5f} recall:{metric["recall"]:.5f} hmean:{metric["hmean"]:.5f}"
            metrics[f"thr {score_thr}"] = metric_str
            hmean = max(hmean, metric["hmean"])
        metrics["hmean"] = hmean

        self.reset()
        return metrics

    def reset(self):
        self.results = {
            0.3: [],
            0.4: [],
            0.5: [],
            0.6: [],
            0.7: [],
            0.8: [],
            0.9: [],
        }  # clear results
