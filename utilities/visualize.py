import cv2 as cv
import numpy as np

from sub_ocr.utils import read_image, rescale, pascal_voc_bb, flatten_iter, crop_image


def get_scale_factor(img_frame: np.ndarray, img_target_height: int = 600) -> float:
    img_height = img_frame.shape[0]
    if img_height > img_target_height:
        rescale_factor = img_target_height / img_height
        return rescale_factor


def display_image(image: np.ndarray, win_name: str, display_time: int = 0) -> None:
    """
    Press any keyboard key to close image.
    """
    cv.imshow(win_name, image)
    cv.waitKey(int(display_time * 1000))
    cv.destroyAllWindows()


def visualize_np_image(image: np.ndarray, title: str, dsp_time: int) -> None:
    if len(image.shape) > 2 and image.shape[-1] > 3:
        image = np.moveaxis(image, 0, -1)  # change image data format from [C, H, W] to [H, W, C]
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    if scale := get_scale_factor(image):
        image, scale = rescale(scale, image), round(scale, 4)
    display_image(image, f"{f'{title} - ' if title else title}Image Rescale Value: {scale}", dsp_time)


def visualize_dataset(dataset, num: int = 5, dsp_time: int = 2) -> None:
    print("Visualizing dataset...")
    ds_len = len(dataset)
    for _ in range(num):
        idx = np.random.randint(ds_len)
        data = dataset[idx]
        print(f"Image Path: {data['image_path']}, Text: {data.get('text')}")
        for key, val in data.items():
            if isinstance(val, np.ndarray):
                visualize_np_image(val, key, dsp_time)


def visualize_data(image_path: str, labels: list, crop_bbox: bool = True, put_text: bool = False) -> None:
    image, image_height, image_width = read_image(image_path, False)  # Load the image
    if scale := get_scale_factor(image):
        image = rescale(scale, image)

    for label in labels:
        bbox, text = label.get("bbox"), label.get("text")
        if bbox:
            bbox = tuple(flatten_iter(bbox))
            bbox = rescale(scale, bbox=bbox) if scale else bbox
            if crop_bbox and len(labels) == 1:
                _, image = crop_image(image, image_height, image_width, bbox)
            else:
                x_min, y_min, x_max, y_max = map(int, pascal_voc_bb(bbox))
                cv.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Draw the bbox on the image
                if text and put_text:
                    cv.putText(image, text, (x_max, y_max), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    display_image(image, f"Image Rescale Value: {round(scale, 4) if scale else scale}")
