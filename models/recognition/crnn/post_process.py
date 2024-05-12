import torch


class StrLabelConverter:
    def __init__(self, alphabet: str, ignore_case: bool = False) -> None:
        """
        Convert between str and label. NOTE: Insert `blank` to the alphabet for CTC.
        :param alphabet: set of the possible characters.
        :param ignore_case: whether to ignore character case.
        """
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text: str | list) -> tuple:
        """
        Support batch or single str.
        :param text: texts to convert.
        :return: torch.LongTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts and
                 torch.LongTensor [n]: length of each text.
        """
        if isinstance(text, str):
            text = [self.dict[char.lower() if self._ignore_case else char] for char in text]
            length = [len(text)]
        else:
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return torch.LongTensor(text), torch.LongTensor(length)

    def decode(self, encoded_txt: torch.Tensor, length: torch.Tensor, raw: bool = False) -> str | list:
        """
        Decode encoded texts back into strs.
        :param encoded_txt: [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
        :param length: length of each text.
        :param raw:
        :return: decoded texts
        Raises: AssertionError: when the texts and its length does not match.
        """
        if length.numel() == 1:
            length = length[0]
            assert encoded_txt.numel() == length, (f"text with length: {encoded_txt.numel()} "
                                                   f"does not match declared length: {length}")
            if raw:
                return ''.join([self.alphabet[i - 1] for i in encoded_txt])
            else:
                char_list = []
                for i in range(length):
                    if encoded_txt[i] != 0 and (not (i > 0 and encoded_txt[i - 1] == encoded_txt[i])):
                        char_list.append(self.alphabet[encoded_txt[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert encoded_txt.numel() == length.sum(), (f"texts with length: {encoded_txt.numel()} "
                                                         f"does not match declared length: {length.sum()}")
            texts = []
            index = 0
            for i in range(length.numel()):
                lx = length[i]
                texts.append(self.decode(encoded_txt[index:index + lx], torch.LongTensor([lx]), raw=raw))
                index += lx
            return texts


class CRNNPostProcess:
    def __init__(self, alphabet: str) -> None:
        self.converter = StrLabelConverter(alphabet)

    def __call__(self, predictions: torch.Tensor) -> tuple:
        prediction_size = torch.LongTensor([predictions.size(0)] * predictions.size(1))
        scores, predictions = predictions.max(2)
        # todo: implement scores for predictions
        predictions = predictions.transpose(1, 0).contiguous().view(-1)
        predictions = self.converter.decode(predictions, prediction_size, False)
        return predictions, scores


if __name__ == '__main__':
    with open("../alphabets/en.txt") as file:
        test_alphabet = "".join([line.rstrip("\n") for line in file])

    test_conv = StrLabelConverter(test_alphabet)
    test_encoded, test_len = test_conv.encode(["Testing 123", "Food", "aaa", "bbb", "ddd", "123", "444"])
    print(test_encoded, test_len)
    print(test_conv.decode(test_encoded, test_len))
