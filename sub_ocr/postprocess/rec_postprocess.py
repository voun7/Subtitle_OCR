import numpy as np
import torch

from sub_ocr.utils import read_chars


class BaseRecLabelDecode:
    """Convert between text-label and text-index"""

    def __init__(self, lang):
        dict_character = list(read_chars(lang))
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
            result_list.append((text, float(np.mean(conf_list)) if conf_list else 0))
        return result_list

    def get_ignored_tokens(self):
        return [0]  # for ctc blank


class CTCLabelDecode(BaseRecLabelDecode):
    """Convert between text-label and text-index"""

    def __init__(self, lang):
        super().__init__(lang)

    def __call__(self, preds):
        preds_prob, preds_idx = preds.max(2)
        preds_prob, preds_idx = preds_prob.transpose(1, 0), preds_idx.transpose(1, 0)
        preds_prob, preds_idx = torch.exp(preds_prob).detach().cpu().numpy(), preds_idx.detach().cpu().numpy()
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        return text

    def add_special_char(self, dict_character):
        dict_character = ["blank"] + dict_character
        return dict_character
