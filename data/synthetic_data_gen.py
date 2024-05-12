import subprocess


def generate_trdg_images() -> None:
    """
    Use trdg package to generate images with text for training model.
    pip install trdg

    Errors and Fixes (paste them directly in the package module)
    ---------------------------------------------------
    getsize() error:
    left, top, right, bottom = image_font.getbbox(text)
    return bottom
    ---------------------------------------------------
    from PIL import Image, ImageFile
    # PIL.Image.DecompressionBombError:
    Image.MAX_IMAGE_PIXELS = None
    # OSError: image file is truncated
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    """
    dataset_dir = r"C:\Users\Victor\Documents\Python Datasets\Subtitle_OCR\TRDG Synthetic Images"
    lang = "en"  # 'en' for English, 'cn' for Chinese (rename folder to ch later)
    command = [
        "trdg",
        "--output_dir", f"{dataset_dir}/{lang}",  # The output directory.
        "--language", lang,  # The language to use.
        "--count", "1000000",  # The number of images to be created.
        "--random",  # Define if the produced string will have variable word count (with --length being the maximum).
        "--thread_count", "18",  # Define the number of thread to use for image generation.
        "--background", "3",  # Background to use. 0: Gaussian Noise, 1: Plain white, 2: Quasi crystal, 3: Image.
        "--image_dir", f"{dataset_dir}/#Background_images",  # Image directory to use when background is set to image.
        "--text_color", "#FFFFFF",  # Text's color. "#000000,#FFFFFF" for black to white range.
        "--name_format", "2",  # Define how the produced files will be named.
        # Rarely needed options.
        # "--font_dir", "data/#Fonts",  # Define a font directory to be used.
        # "--word_split",  # Split on words instead of on characters.
    ]
    if lang == "en":
        command.extend([
            "--length", "2",  # Define how many words should be included in each generated sample.
        ])
    elif lang == "cn":
        command.extend([
            "--length", "18",  # Define how many words should be included in each generated sample.
            "--space_width", "0",  # Define the width of the spaces between words
        ])
    print(f"Command: {' '.join(command)}")
    try:
        # Run the command using subprocess.run().
        subprocess.run(command)
    except Exception as error:
        print(f"An error occurred. Error: {error}")


if __name__ == '__main__':
    generate_trdg_images()
