from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2 as cv
import numpy as np


def check_images(blank_image_dir: Path, image_files: list) -> None:
    for image_file in image_files:
        image = cv.imread(str(image_file))
        image_mean = np.mean(image)
        if image_mean < 5 or image_mean > 245:  # threshold for mostly black and mostly white images
            image_file.rename(blank_image_dir / image_file.name)
            print(f"Image: {image_file.name}, Mean: {image_mean} moved to blank images dir")


def blank_image_cleanup(image_dir: Path) -> None:
    """
    Remove blank images from directory.
    """
    blank_image_dir, chunk_size, all_image_files = image_dir / "blank images", 400, list(image_dir.glob("*.jpg"))
    blank_image_dir.mkdir(exist_ok=True)
    image_file_chunks = [all_image_files[i:i + chunk_size] for i in range(0, len(all_image_files), chunk_size)]
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(check_images, blank_image_dir, image_files) for image_files in image_file_chunks]
        for f in futures:
            f.result()  # Prevents silent bugs. Exceptions raised will be displayed.


if __name__ == '__main__':
    blank_image_cleanup(Path(r""))
