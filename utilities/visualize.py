import cv2 as cv
import numpy as np

from utilities.utils import rescale, pascal_voc_bb, flatten_iter


def get_scale_factor(img_frame: np.ndarray, img_target_height: int = 700) -> float:
    img_height = img_frame.shape[0]
    if img_height > img_target_height:
        rescale_factor = img_target_height / img_height
        return rescale_factor


def display_image(image: np.ndarray, win_name: str) -> None:
    """
    Press any keyboard key to close image.
    """
    cv.imshow(win_name, image)
    cv.waitKey()
    cv.destroyAllWindows()


def visualize_np_image(image: np.ndarray, title: str = None) -> None:
    if len(image.shape) > 2 and image.shape[-1] > 3:
        image = np.moveaxis(image, 0, -1)  # change image data format from [C, H, W] to [H, W, C]
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    if scale := get_scale_factor(image):
        image = rescale(scale, image)
    title = f"{title} - " or ""
    display_image(image, f"{title}Image Rescale Value: {round(scale, 4) if scale else scale}")


def visualize_dataset(dataset, num: int = 50) -> None:
    ds_len = len(dataset)
    for _ in range(num):
        idx = np.random.randint(ds_len)
        data = dataset[idx]
        print(f"Image Path: {data['image_path']}, Text: {data.get('text')}")
        for key, val in data.items():
            if isinstance(val, np.ndarray):
                visualize_np_image(val, key)


def visualize_data(image: str, labels: list, crop_bbox: bool = True, put_text: bool = False) -> None:
    image = cv.imread(image)  # Load the image
    if scale := get_scale_factor(image):
        image = rescale(scale, image)

    for label in labels:
        bbox, text = label["bbox"], label["text"]
        if bbox:
            bbox = tuple(flatten_iter(bbox))
            bbox = rescale(scale, bbox=bbox) if scale else bbox
            x_min, y_min, x_max, y_max = map(int, pascal_voc_bb(bbox))  # Change type to int and change bbox format
            if crop_bbox and len(labels) == 1:
                image = image[y_min:y_max, x_min:x_max]
            else:
                cv.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Draw the bbox on the image
            if text and put_text:
                cv.putText(image, text, (x_max, y_max), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    display_image(image, f"Image Rescale Value: {round(scale, 6) if scale else scale}")
