import copy

import numpy as np

from sub_ocr.utils import read_chars


class BaseRecLabelEncode:
    """Convert between text-label and text-index"""

    def __init__(self, lang, max_text_length):
        self.max_text_len = max_text_length
        self.beg_str = "sos"
        self.end_str = "eos"

        self.character_str = read_chars(lang)
        dict_character = list(self.character_str)
        dict_character = self.add_special_char(dict_character)
        self.dict = {character: index for index, character in enumerate(dict_character)}
        self.character = dict_character

    def add_special_char(self, dict_character):
        return dict_character

    def encode(self, text):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]

        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        if len(text) == 0 or len(text) > self.max_text_len:
            return None
        text_list = []
        for char in text:
            if char not in self.dict:
                continue
            text_list.append(self.dict[char])
        if len(text_list) == 0:
            return None
        return text_list


class CTCLabelEncode(BaseRecLabelEncode):
    """Convert between text-label and text-index"""

    def __init__(self, lang, max_text_length):
        super().__init__(lang, max_text_length)

    def __call__(self, data):
        text = data["label"]
        text = self.encode(text)
        if text is None:
            return None
        data["length"] = np.array(len(text))
        text = text + [0] * (self.max_text_len - len(text))
        data["label"] = np.array(text)

        label = [0] * len(self.character)
        for x in text:
            label[x] += 1
        data["label_ace"] = np.array(label)
        return data

    def add_special_char(self, dict_character):
        dict_character = ["blank"] + dict_character
        return dict_character


class AttnLabelEncode(BaseRecLabelEncode):
    """Convert between text-label and text-index"""

    def __init__(self, lang, max_text_length):
        super().__init__(lang, max_text_length)

    def add_special_char(self, dict_character):
        self.beg_str = "sos"
        self.end_str = "eos"
        dict_character = [self.beg_str] + dict_character + [self.end_str]
        return dict_character

    def __call__(self, data):
        text = data["label"]
        text = self.encode(text)
        if text is None:
            return None
        if len(text) >= self.max_text_len:
            return None
        data["length"] = np.array(len(text))
        text = ([0] + text + [len(self.character) - 1] + [0] * (self.max_text_len - len(text) - 2))
        data["label"] = np.array(text)
        return data

    def get_ignored_tokens(self):
        beg_idx = self.get_beg_end_flag_idx("beg")
        end_idx = self.get_beg_end_flag_idx("end")
        return [beg_idx, end_idx]

    def get_beg_end_flag_idx(self, beg_or_end):
        if beg_or_end == "beg":
            idx = np.array(self.dict[self.beg_str])
        elif beg_or_end == "end":
            idx = np.array(self.dict[self.end_str])
        else:
            assert False, "Unsupport type %s in get_beg_end_flag_idx" % beg_or_end
        return idx


class SARLabelEncode(BaseRecLabelEncode):
    """Convert between text-label and text-index"""

    def __init__(self, lang, max_text_length):
        super().__init__(lang, max_text_length)

    def add_special_char(self, dict_character):
        beg_end_str = "<BOS/EOS>"
        unknown_str = "<UKN>"
        padding_str = "<PAD>"
        dict_character = dict_character + [unknown_str]
        self.unknown_idx = len(dict_character) - 1
        dict_character = dict_character + [beg_end_str]
        self.start_idx = len(dict_character) - 1
        self.end_idx = len(dict_character) - 1
        dict_character = dict_character + [padding_str]
        self.padding_idx = len(dict_character) - 1

        return dict_character

    def __call__(self, data):
        text = data["label"]
        text = self.encode(text)
        if text is None:
            return None
        if len(text) >= self.max_text_len - 1:
            return None
        data["length"] = np.array(len(text))
        target = [self.start_idx] + text + [self.end_idx]
        padded_text = [self.padding_idx for _ in range(self.max_text_len)]

        padded_text[: len(target)] = target
        data["label"] = np.array(padded_text)
        return data

    def get_ignored_tokens(self):
        return [self.padding_idx]


class MultiLabelEncode(BaseRecLabelEncode):
    def __init__(self, lang, max_text_length, gtc_encode=None):
        super().__init__(lang, max_text_length)

        self.ctc_encode = CTCLabelEncode(lang, max_text_length)
        self.gtc_encode_type = gtc_encode
        if gtc_encode is None:
            self.gtc_encode = SARLabelEncode(lang, max_text_length)
        else:
            self.gtc_encode = eval(gtc_encode)(lang, max_text_length)

    def __call__(self, data):
        data_ctc = copy.deepcopy(data)
        data_gtc = copy.deepcopy(data)
        data_out = dict()
        data_out["img_path"] = data.get("img_path", None)
        data_out["image"] = data["image"]
        ctc = self.ctc_encode.__call__(data_ctc)
        gtc = self.gtc_encode.__call__(data_gtc)
        if ctc is None or gtc is None:
            return None
        data_out["label_ctc"] = ctc["label"]
        if self.gtc_encode_type is not None:
            data_out["label_gtc"] = gtc["label"]
        else:
            data_out["label_sar"] = gtc["label"]
        data_out["length"] = ctc["length"]
        return data_out


class NRTRLabelEncode(BaseRecLabelEncode):
    """Convert between text-label and text-index"""

    def __init__(self, lang, max_text_length):
        super().__init__(lang, max_text_length)

    def __call__(self, data):
        text = data["label"]
        text = self.encode(text)
        if text is None:
            return None
        if len(text) >= self.max_text_len - 1:
            return None
        data["length"] = np.array(len(text))
        text.insert(0, 2)
        text.append(3)
        text = text + [0] * (self.max_text_len - len(text))
        data["label"] = np.array(text)
        return data

    def add_special_char(self, dict_character):
        dict_character = ["blank", "<unk>", "<s>", "</s>"] + dict_character
        return dict_character


class ViTSTRLabelEncode(BaseRecLabelEncode):
    """Convert between text-label and text-index"""

    def __init__(self, lang, max_text_length, ignore_index=0):
        super().__init__(lang, max_text_length)
        self.ignore_index = ignore_index

    def __call__(self, data):
        text = data["label"]
        text = self.encode(text)
        if text is None:
            return None
        if len(text) >= self.max_text_len:
            return None
        data["length"] = np.array(len(text))
        text.insert(0, self.ignore_index)
        text.append(1)
        text = text + [self.ignore_index] * (self.max_text_len + 2 - len(text))
        data["label"] = np.array(text)
        return data

    def add_special_char(self, dict_character):
        dict_character = ["<s>", "</s>"] + dict_character
        return dict_character


class CANLabelEncode(BaseRecLabelEncode):
    def __init__(self, lang, max_text_length=100):
        super().__init__(lang, max_text_length)

    def encode(self, text_seq):
        text_seq_encoded = []
        for text in text_seq:
            if text not in self.character:
                continue
            text_seq_encoded.append(self.dict.get(text))
        if len(text_seq_encoded) == 0:
            return None
        return text_seq_encoded

    def __call__(self, data):
        label = data["label"]
        if isinstance(label, str):
            label = label.strip().split()
        label.append(self.end_str)
        data["label"] = self.encode(label)
        return data
