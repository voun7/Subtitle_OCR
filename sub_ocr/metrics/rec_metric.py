import string

from rapidfuzz.distance import Levenshtein


class RecMetric:
    def __init__(self, post_processor, is_filter=False, ignore_space=True):
        self.post_processor, self.is_filter, self.ignore_space = post_processor, is_filter, ignore_space

    def _normalize_text(self, text):
        text = "".join(filter(lambda x: x in (string.digits + string.ascii_letters), text))
        return text.lower()

    def __call__(self, predictions, batch, _):
        predictions = self.post_processor(predictions)
        correct_num = all_num = norm_edit_dis = 0
        for (prediction, _), (target) in zip(predictions, batch["text"]):
            if self.ignore_space:
                prediction, target = prediction.replace(" ", ""), target.replace(" ", "")
            if self.is_filter:
                prediction, target = self._normalize_text(prediction), self._normalize_text(target)
            norm_edit_dis += Levenshtein.normalized_distance(prediction, target)
            if prediction == target:
                correct_num += 1
            all_num += 1
        correct_num += correct_num
        all_num += all_num
        norm_edit_dis += norm_edit_dis
        return {"acc": correct_num / all_num, "norm_edit_dis": 1 - norm_edit_dis / all_num}
