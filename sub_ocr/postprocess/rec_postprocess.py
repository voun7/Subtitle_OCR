import numpy as np
import torch

from sub_ocr.utils import read_chars


class BaseRecLabelDecode:
    """Convert between text-label and text-index"""

    def __init__(self, lang, **kwargs):
        self.beg_str = "sos"
        self.end_str = "eos"
        self.character_str = []

        self.character_str = read_chars(lang)
        dict_character = list(self.character_str)
        dict_character = self.add_special_char(dict_character)
        self.dict = {character: index for index, character in enumerate(dict_character)}
        self.character = dict_character

    def add_special_char(self, dict_character):
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if is_remove_duplicate:
                    # only for predict
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[batch_idx][idx]:
                        continue
                char_list.append(self.character[int(text_index[batch_idx][idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list) if conf_list else 0))
        return result_list

    def get_ignored_tokens(self):
        return [0]  # for ctc blank


class CTCLabelDecode(BaseRecLabelDecode):
    """Convert between text-label and text-index"""

    def __init__(self, lang, **kwargs):
        super().__init__(lang, **kwargs)

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)

        if label is None:
            return text
        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ["blank"] + dict_character
        return dict_character
