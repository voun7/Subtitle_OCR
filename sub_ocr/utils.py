import math
from pathlib import Path
from typing import NewType, Iterable, Generator

import cv2 as cv
import numpy as np


class Types:
    ModelType = NewType('ModelType', str)
    ModelName = NewType('ModelName', str)
    DataType = NewType('DataType', str)
    Language = NewType('Language', str)

    det = ModelType("Detection")
    rec = ModelType("Recognition")

    db = ModelName("DB")
    crnn = ModelName("CRNN")
    svtr = ModelName("SVTR")

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
        return cv.resize(frame, None, fx=scale, fy=scale)

    if bbox:
        return tuple(map(lambda c: c * scale, bbox))


def bbox_area(x_min: int, y_min: int, x_max: int, y_max: int) -> float:
    return (x_max - x_min) * (y_max - y_min)


def crop_image(image: np.ndarray, image_height: int, image_width: int, bbox: tuple) -> tuple:
    """
    Crop image using bbox. If cropping fails a blank image will be created with the height and width.
    """
    blank_image, bbox = False, pascal_voc_bb(tuple(flatten_iter(bbox)))
    x_min, y_min, x_max, y_max = map(int, bbox)
    if not bbox_area(x_min, y_min, x_max, y_max):
        x_min, y_min, x_max, y_max = map(round, bbox)
    # the bbox coordinates will not be allowed to be out of bounds or negative.
    x_min, y_min = max(min(x_min, image_width), 1), max(min(y_min, image_height), 1)
    x_max, y_max = max(min(x_max, image_width), 1), max(min(y_max, image_height), 1)
    cropped_image = image[y_min:y_max, x_min:x_max]  # crop image with bbox.
    if not cropped_image.size:  # blank image will be used if crop fails.
        blank_image, cropped_image = True, np.zeros((image_height, image_width, 3), np.uint8)  # blank image
    return blank_image, cropped_image


def resize_norm_img(image: np.ndarray, target_height: int, target_width: int, pad: bool = True) -> tuple:
    """
    Image scaling and normalization
    :return: resized normalized image and the rescale value
    """
    # Calculate the scaling factor to resize the image
    scale = min(target_height / image.shape[0], target_width / image.shape[1])
    # Resize the image while maintaining aspect ratio
    resized_image = rescale(scale, image)
    if pad:  # Add padding if requested
        pad_h, pad_w = target_height - resized_image.shape[0], target_width - resized_image.shape[1]
        resized_image = cv.copyMakeBorder(resized_image, 0, pad_h, 0, pad_w, cv.BORDER_CONSTANT, value=(0, 0, 0))
    normalized_image = cv.normalize(resized_image, None, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    image = np.moveaxis(normalized_image, -1, 0)  # change image data format from [H, W, C] to [C, H, W]
    return image, scale


def read_chars(lang: Types.Language) -> str:
    """
    Read the alphabet of the given language from the language file.
    """
    alphabet_file = Path(__file__).parent / f"models/recognition/alphabets/{lang}.txt"
    alphabet = " " + "".join([line.rstrip("\n") for line in alphabet_file.read_text(encoding="utf-8")])
    return alphabet


def rect_corners(x: float, y: float, width: float, height: float, theta: float) -> tuple:
    """
    Use given args to generate rectangle corners along with the rotation.
    """

    def xy_rotate(theta_: float, x_: float, y_: float, cx_: float, cy_: float) -> tuple:
        rotated_x = math.cos(theta_) * (x_ - cx_) - math.sin(theta_) * (y_ - cy_)
        rotated_y = math.cos(theta_) * (y_ - cy_) + math.sin(theta_) * (x_ - cx_)
        return cx_ + rotated_x, cy_ + rotated_y

    cx, cy = x + width / 2, y + height / 2
    x1, y1 = xy_rotate(theta, x, y, cx, cy)
    x2, y2 = xy_rotate(theta, x + width, y, cx, cy)
    x3, y3 = xy_rotate(theta, x, y + height, cx, cy)
    x4, y4 = xy_rotate(theta, x + width, y + height, cx, cy)
    return (x1, y1), (x3, y3), (x4, y4), (x2, y2)
