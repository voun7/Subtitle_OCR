import subprocess
from os import cpu_count


def generate_trdg_images(lang: str = "ch", count: int = 4_000_000) -> None:
    """
    Use trdg package to generate synthetic images with text for training model.
    Cleanup module should be used for deleting all images that have blended with the background.
    python.exe -m pip install pip==24.0
    pip install git+https://github.com/Belval/TextRecognitionDataGenerator.git (pip<24.1 required)

    Errors and Fixes (paste them directly in the package module)
    ---------------------------------------------------
    from PIL import Image, ImageFile
    # PIL.Image.DecompressionBombError:
    Image.MAX_IMAGE_PIXELS = None
    # OSError: image file is truncated
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    """
    dataset_dir = r"C:\Users\Victor\Documents\Python Datasets\Subtitle_OCR\TRDG Synthetic Images"
    command = [
        "trdg",
        "--output_dir", f"{dataset_dir}/{lang}",  # The output directory.
        "--count", str(count),  # The number of images to be created.
        "--random",  # Define if the produced string will have variable word count (with --length being the maximum).
        "--thread_count", str(cpu_count()),  # Define the number of thread to use for image generation.
        "--background", "3",  # Background to use. 0: Gaussian Noise, 1: Plain white, 2: Quasi crystal, 3: Image.
        "--image_dir", f"{dataset_dir}/#Background_images",  # Image directory to use when background is set to image.
        "--text_color", "#FFFFFF",  # Text's color. "#000000,#FFFFFF" for black to white range.
        "--name_format", "2",  # Define how the produced files will be named.
        "--format", "100",  # Define the height of the produced images if horizontal, else the width.
        # Rarely needed options.
        # "--font_dir", "data/#Fonts",  # Define a font directory to be used.
        # "--word_split",  # Split on words instead of on characters.
    ]
    if lang == "sb":
        command.extend(["--random_sequences"])  # Use random sequences as the source text for the generation.
        lang = "en"

    if lang == "en":
        command.extend([
            "--language", lang,  # The language to use.
            "--length", "2",  # Define how many words should be included in each generated sample.
            "--margins", "5,10,15,10",
        ])
    elif lang == "ch":
        command.extend([
            "--language", "cn",  # The language to use. (trdg uses cn for chinese)
            "--length", "12",  # Define how many characters should be included in each generated sample.
            "--space_width", "0",  # Define the width of the spaces between words
            "--margins", "0,10,20,10",
        ])
    print(f"Command: {' '.join(command)}")
    try:
        # Run the command using subprocess.run().
        subprocess.run(command)
    except Exception as error:
        print(f"An error occurred. Error: {error}")


if __name__ == '__main__':
    generate_trdg_images()
