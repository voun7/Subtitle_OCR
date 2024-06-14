import cv2 as cv
import numpy as np
from shapely.geometry import Polygon


def iou_rotate(box_a, box_b, method='union'):
    rect_a = cv.minAreaRect(box_a)
    rect_b = cv.minAreaRect(box_b)
    r1 = cv.rotatedRectangleIntersection(rect_a, rect_b)
    if r1[0] == 0:
        return 0
    else:
        inter_area = cv.contourArea(r1[1])
        area_a = cv.contourArea(box_a)
        area_b = cv.contourArea(box_b)
        union_area = area_a + area_b - inter_area

        if union_area == 0 or inter_area == 0:
            return 0
        if method == 'union':
            iou = inter_area / union_area
        elif method == 'intersection':
            iou = inter_area / min(area_a, area_b)
        else:
            raise NotImplementedError
        return iou


class DetectionIoUEvaluator:
    def __init__(self, iou_constraint=0.5, area_precision_constraint=0.5, is_output_polygon=False):
        self.is_output_polygon = is_output_polygon
        self.iou_constraint = iou_constraint
        self.area_precision_constraint = area_precision_constraint

    def evaluate_image(self, gt, prediction):

        def get_union(pD, pG):
            return Polygon(pD).union(Polygon(pG)).area

        def get_intersection_over_union(pD, pG):
            return get_intersection(pD, pG) / get_union(pD, pG)

        def get_intersection(pD, pG):
            return Polygon(pD).intersection(Polygon(pG)).area

        def compute_ap(conf_list, match_list, num_gt_care):
            correct = 0
            AP = 0
            if len(conf_list) > 0:
                conf_list = np.array(conf_list)
                match_list = np.array(match_list)
                sorted_ind = np.argsort(-conf_list)
                conf_list = conf_list[sorted_ind]
                match_list = match_list[sorted_ind]
                for n in range(len(conf_list)):
                    match = match_list[n]
                    if match:
                        correct += 1
                        AP += float(correct) / (n + 1)

                if num_gt_care > 0:
                    AP /= num_gt_care
            return AP

        matched_sum = 0

        num_global_care_gt = 0
        num_global_care_det = 0

        det_matched = 0

        iou_mat = np.empty([1, 1])

        gt_pols = []
        det_pols = []

        gt_pol_points = []
        det_pol_points = []

        pairs = []
        det_matched_nums = []

        evaluation_log = ""

        for n in range(len(gt)):
            points = gt[n]['bbox']

            if not Polygon(points).is_valid or not Polygon(points).is_simple:
                continue

            gt_pol = points
            gt_pols.append(gt_pol)
            gt_pol_points.append(points)

        evaluation_log += "GT polygons: " + str(len(gt_pols))

        for n in range(len(prediction)):
            points = prediction[n]['bbox']

            if not Polygon(points).is_valid or not Polygon(points).is_simple:
                continue

            det_pol = points
            det_pols.append(det_pol)
            det_pol_points.append(points)

        evaluation_log += " DET polygons: " + str(len(det_pols))

        if len(gt_pols) > 0 and len(det_pols) > 0:
            # Calculate IoU and precision matrix's
            output_shape = [len(gt_pols), len(det_pols)]
            iou_mat = np.empty(output_shape)
            gt_rect_mat = np.zeros(len(gt_pols), np.int8)
            det_rect_mat = np.zeros(len(det_pols), np.int8)
            if self.is_output_polygon:
                for gtNum in range(len(gt_pols)):
                    for detNum in range(len(det_pols)):
                        pG = gt_pols[gtNum]
                        pD = det_pols[detNum]
                        iou_mat[gtNum, detNum] = get_intersection_over_union(pD, pG)
            else:
                for gtNum in range(len(gt_pols)):
                    for detNum in range(len(det_pols)):
                        pG = np.float32(gt_pols[gtNum])
                        pD = np.float32(det_pols[detNum])
                        iou_mat[gtNum, detNum] = iou_rotate(pD, pG)
            for gtNum in range(len(gt_pols)):
                for detNum in range(len(det_pols)):
                    if gt_rect_mat[gtNum] == 0 and det_rect_mat[detNum] == 0:
                        if iou_mat[gtNum, detNum] > self.iou_constraint:
                            gt_rect_mat[gtNum] = 1
                            det_rect_mat[detNum] = 1
                            det_matched += 1
                            pairs.append({'gt': gtNum, 'det': detNum})
                            det_matched_nums.append(detNum)
                            evaluation_log += " Match GT #" + str(gtNum) + " with Det #" + str(detNum)

        num_gt_care = len(gt_pols)
        num_det_care = len(det_pols)
        if num_gt_care == 0:
            recall = float(1)
            precision = float(0) if num_det_care > 0 else float(1)
        else:
            recall = float(det_matched) / num_gt_care
            precision = 0 if num_det_care == 0 else float(det_matched) / num_det_care

        h_mean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)

        matched_sum += det_matched
        num_global_care_gt += num_gt_care
        num_global_care_det += num_det_care

        per_sample_metrics = {
            'precision': precision,
            'recall': recall,
            'h_mean': h_mean,
            'pairs': pairs,
            'iou_mat': [] if len(det_pols) > 100 else iou_mat.tolist(),
            'gt_pol_points': gt_pol_points,
            'det_pol_points': det_pol_points,
            'gtCare': num_gt_care,
            'detCare': num_det_care,
            'det_matched': det_matched,
            'evaluation_log': evaluation_log
        }
        return per_sample_metrics

    @staticmethod
    def combine_results(results):
        num_global_care_gt = 0
        num_global_care_det = 0
        matched_sum = 0
        for result in results:
            num_global_care_gt += result['gtCare']
            num_global_care_det += result['detCare']
            matched_sum += result['det_matched']

        method_recall = 0 if num_global_care_gt == 0 else float(matched_sum) / num_global_care_gt
        method_precision = 0 if num_global_care_det == 0 else float(matched_sum) / num_global_care_det
        method_h_mean = 0 if method_recall + method_precision == 0 else 2 * method_recall * method_precision / (
                method_recall + method_precision)

        method_metrics = {'precision': method_precision, 'recall': method_recall, 'h_mean': method_h_mean}
        return method_metrics


if __name__ == '__main__':
    test_evaluator = DetectionIoUEvaluator()
    test_predictions = [[{
        'bbox': [(0.1, 0.1), (0.5, 0), (0.5, 1), (0, 1)],
        'text': 1234,
    }, {
        'bbox': [(0.5, 0.1), (1, 0), (1, 1), (0.5, 1)],
        'text': 5678,
    }]]
    test_gts = [[{
        'bbox': [(0.1, 0.1), (1, 0), (1, 1), (0, 1)],
        'text': 123,
    }]]
    test_results = []
    for gt, predictions in zip(test_gts, test_predictions):
        test_results.append(test_evaluator.evaluate_image(gt, predictions))
    test_metrics = test_evaluator.combine_results(test_results)
    print(test_metrics)
