import copy
from unittest import TestCase

from data.preprocess import rec_label_ops as enc
from sub_ocr.postprocess import rec_postprocess as dec


class TestRecDataEncodeDecode(TestCase):
    def test_encoding_decoding(self) -> None:
        lang, mx_txt_len = "en", 25
        test_text = {"label": "This is @ Sample Text."}
        enc_decs = ["CTCLabel", "AttnLabel", "NRTRLabel", "ViTSTRLabel"]

        print("\nTesting encoding and decoding of labels...")
        for i, name in enumerate(enc_decs):
            with self.subTest(f"{name}", i=i):
                encoder, decoder = getattr(enc, f"{name}Encode")(lang, mx_txt_len), getattr(dec, f"{name}Decode")(lang)
                encoded_text = encoder(copy.deepcopy(test_text))
                decoded_text = decoder.decode([encoded_text["label"]])
                decoded_text = decoded_text[0][0].replace("<s>", "")
                self.assertEqual(test_text["label"], decoded_text)
                print(f"{name} passed test...")
