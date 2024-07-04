import numpy as np
from shapely.geometry import Polygon

"""
reference from :
https://github.com/MhLiao/DB/blob/3c32b808d4412680310d3d28eeb6a2d5bf1566c5/concern/icdar2015_eval/detection/iou.py#L8
"""


class DetectionIoUEvaluator:
    def __init__(self, iou_constraint=0.5, area_precision_constraint=0.5):
        self.iou_constraint = iou_constraint
        self.area_precision_constraint = area_precision_constraint

    def evaluate_image(self, gt, pred):
        def get_union(pD, pG):
            return Polygon(pD).union(Polygon(pG)).area

        def get_intersection_over_union(pD, pG):
            return get_intersection(pD, pG) / get_union(pD, pG)

        def get_intersection(pD, pG):
            return Polygon(pD).intersection(Polygon(pG)).area

        matchedSum = 0

        numGlobalCareGt = 0
        numGlobalCareDet = 0

        detMatched = 0

        gtPols = []
        detPols = []

        gtPolPoints = []
        detPolPoints = []

        pairs = []
        detMatchedNums = []

        evaluationLog = ""

        for n in range(len(gt)):
            points = gt[n]["bbox"]
            if not Polygon(points).is_valid:
                continue

            gtPol = points
            gtPols.append(gtPol)
            gtPolPoints.append(points)

        evaluationLog += "GT polygons: " + str(len(gtPols))

        for n in range(len(pred)):
            points = pred[n]["bbox"]
            if not Polygon(points).is_valid:
                continue

            detPol = points
            detPols.append(detPol)
            detPolPoints.append(points)

        evaluationLog += "DET polygons: " + str(len(detPols))

        if len(gtPols) > 0 and len(detPols) > 0:
            # Calculate IoU and precision matrix's
            outputShape = [len(gtPols), len(detPols)]
            iouMat = np.empty(outputShape)
            gtRectMat = np.zeros(len(gtPols), np.int8)
            detRectMat = np.zeros(len(detPols), np.int8)
            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    pG = gtPols[gtNum]
                    pD = detPols[detNum]
                    iouMat[gtNum, detNum] = get_intersection_over_union(pD, pG)

            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0:
                        if iouMat[gtNum, detNum] > self.iou_constraint:
                            gtRectMat[gtNum] = 1
                            detRectMat[detNum] = 1
                            detMatched += 1
                            pairs.append({"gt": gtNum, "det": detNum})
                            detMatchedNums.append(detNum)
                            evaluationLog += "Match GT #" + str(gtNum) + " with Det #" + str(detNum)

        numGtCare = len(gtPols)
        numDetCare = len(detPols)

        matchedSum += detMatched
        numGlobalCareGt += numGtCare
        numGlobalCareDet += numDetCare

        perSampleMetrics = {"gtCare": numGtCare, "detCare": numDetCare, "detMatched": detMatched, }
        return perSampleMetrics

    def combine_results(self, results):
        numGlobalCareGt = 0
        numGlobalCareDet = 0
        matchedSum = 0
        for result in results:
            numGlobalCareGt += result["gtCare"]
            numGlobalCareDet += result["detCare"]
            matchedSum += result["detMatched"]

        methodRecall = (0 if numGlobalCareGt == 0 else float(matchedSum) / numGlobalCareGt)
        methodPrecision = (0 if numGlobalCareDet == 0 else float(matchedSum) / numGlobalCareDet)
        methodHmean = (0 if methodRecall + methodPrecision == 0 else 2 * methodRecall * methodPrecision / (
                methodRecall + methodPrecision))
        methodMetrics = {"precision": methodPrecision, "recall": methodRecall, "hmean": methodHmean}
        return methodMetrics


if __name__ == "__main__":
    test_evaluator = DetectionIoUEvaluator()
    test_predictions = [[{'bbox': [(0.1, 0.1), (0.5, 0), (0.5, 1), (0, 1)], 'text': 1234},
                         {'bbox': [(0.5, 0.1), (1, 0), (1, 1), (0.5, 1)], 'text': 5678}]]
    test_gts = [[{'bbox': [(0.1, 0.1), (1, 0), (1, 1), (0, 1)], 'text': 123}]]
    test_results = []
    for gt, predictions in zip(test_gts, test_predictions):
        test_results.append(test_evaluator.evaluate_image(gt, predictions))
    test_metrics = test_evaluator.combine_results(test_results)
    print(test_metrics)
