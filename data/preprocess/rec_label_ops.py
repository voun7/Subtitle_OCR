import numpy as np

from sub_ocr.utils import read_chars


class BaseRecLabelEncode:
    """Convert between text-label and text-index"""

    def __init__(self, lang: str, max_text_length: int) -> None:
        self.max_text_len = max_text_length

        self.character_str = read_chars(lang)
        dict_character = list(self.character_str)
        dict_character = self.add_special_char(dict_character)
        self.dict = {character: index for index, character in enumerate(dict_character)}
        self.character = dict_character

    def add_special_char(self, dict_character: list) -> list:
        return dict_character

    def encode(self, text: str) -> list | None:
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]

        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        if len(text) == 0 or len(text) > self.max_text_len:
            return
        text_list = []
        for char in text:
            if char not in self.dict:
                continue
            text_list.append(self.dict[char])
        if len(text_list) == 0:
            return
        return text_list


class CTCLabelEncode(BaseRecLabelEncode):
    """Convert between text-label and text-index"""

    def __init__(self, lang: str, max_text_length: int) -> None:
        super().__init__(lang, max_text_length)

    def __call__(self, data: dict) -> dict | None:
        text = self.encode(data["text"])
        if text is None:
            return
        data["length"] = np.array(len(text))
        text = text + [0] * (self.max_text_len - len(text))
        data["label"] = np.array(text)
        return data

    def add_special_char(self, dict_character: list) -> list:
        dict_character = ["blank"] + dict_character
        return dict_character
