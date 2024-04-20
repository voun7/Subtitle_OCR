from typing import NewType, Iterable, Generator

import cv2 as cv


class Types:
    ModelType = NewType('ModelType', str)
    ModelName = NewType('ModelName', str)
    DataType = NewType('DataType', str)
    Language = NewType('Language', str)

    det = ModelType("detection")
    rec = ModelType("recognition")

    db = ModelName("DB")
    db_pp = ModelName("DB++")
    crnn = ModelName("CRNN")

    train = DataType("train")  # Training
    val = DataType("val")  # Validation

    english = Language("en")
    chinese = Language("ch")


def pairwise_tuples(data):
    return tuple(zip(data[::2], data[1::2]))


def flatten_iter(iterable: Iterable) -> Generator:
    """
    Function used for removing nested iterables in python using recursion.
    """
    for iter_ in iterable:
        if isinstance(iter_, Iterable) and not isinstance(iter_, (str, bytes)):
            for iter_var in flatten_iter(iter_):
                yield iter_var
        else:
            yield iter_


def pascal_voc_bb(bbox: tuple) -> tuple:
    """
    pascal_voc is a format used by the Pascal VOC dataset. Coordinates of a bounding box are encoded with four
    values in pixels: [x_min, y_min, x_max, y_max]. x_min and y_min are coordinates of the top-left corner of
    the bounding box. x_max and y_max are coordinates of bottom-right corner of the bounding box.
    :param bbox: bbox with eight values representing x1,y1,x2,y2,x3,y3,x4,y4.
    :return: x_min, y_min, x_max, y_max
    """
    x_values, y_values = bbox[::2], bbox[1::2]
    return min(x_values), min(y_values), max(x_values), max(y_values)


def read_image(image_path: str, rgb: bool = True) -> tuple:
    """
    Read image with opencv and change color from bgr to rgb.
    :param image_path: image file location.
    :param rgb: The color format be will be changed to rgb.
    :return: image, image_height, image width
    """
    image = cv.imread(image_path)
    if rgb:
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image_height, image_width, _ = image.shape
    return image, image_height, image_width
