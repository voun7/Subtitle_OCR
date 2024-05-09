from typing import NewType, Iterable, Generator

import cv2 as cv
import numpy as np


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


def rescale(scale: float, frame: np.ndarray = None, bbox: tuple = None) -> np.ndarray | tuple:
    """
    Method to rescale any image frame or bbox using scale.
    Bbox is returned as an integer. This function should be used only for visualization.
    """
    if frame is not None:
        width, height = int(frame.shape[1] * scale), int(frame.shape[0] * scale)
        return cv.resize(frame, (width, height))

    if bbox:
        return tuple(map(lambda c: c * scale, bbox))


def resize_norm_img(image: np.ndarray, new_height: int, new_width: int, pad: bool = True) -> tuple:
    """
    Image scaling and normalization
    :return: resized normalized image and the rescale value
    """
    # Get the dimensions of the original image
    original_height, original_width = image.shape[:2]
    # Calculate aspect ratios
    original_aspect_ratio, new_aspect_ratio = original_width / original_height, new_width / new_height

    # Calculate scaling factor
    if original_aspect_ratio > new_aspect_ratio:
        scaling_factor = new_width / original_width
    else:
        scaling_factor = new_height / original_height

    # Resize the image while maintaining aspect ratio
    resized_image = cv.resize(image, (int(original_width * scaling_factor), int(original_height * scaling_factor)))

    if pad:  # Add padding if requested
        pad_height, pad_width = new_height - resized_image.shape[0], new_width - resized_image.shape[1]
        bottom_pad, right_pad = pad_height, pad_width
        resized_image = cv.copyMakeBorder(resized_image, 0, bottom_pad, 0, right_pad, cv.BORDER_CONSTANT,
                                          value=(0, 0, 0))

    normalized_image = cv.normalize(resized_image, None, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    image = np.moveaxis(normalized_image, -1, 0)  # change image data format from [H, W, C] to [C, H, W]
    return image, scaling_factor
