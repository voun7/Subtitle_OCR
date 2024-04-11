from typing import NewType


class Types:
    ModelType = NewType('ModelType', str)
    DataType = NewType('DataType', str)
    Language = NewType('Language', str)

    det = ModelType("detection")
    rec = ModelType("recognition")

    train = DataType("train")  # Training
    val = DataType("val")  # Validation

    english = Language("en")
    chinese = Language("ch")


def calc_mean_std():
    pass
