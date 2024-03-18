import subprocess


def generate_trdg_images() -> None:
    """
    Use trdg package to generate images with text for training model.
    """
    dataset_dir = r"C:\Users\Victor\Documents\Python Datasets\Subtitle_OCR"
    lang = "en"
    command = [
        "trdg",
        "--output_dir", f"{dataset_dir}/{lang}/rec/synthetic images",  # The output directory.
        "--language", lang,  # The language to use.
        "--count", "1000",  # The number of images to be created.
        "--length", "18",  # Define how many words should be included in each generated sample.
        "--random",  # Define if the produced string will have variable word count (with --length being the maximum).
        "--thread_count", "15",  # Define the number of thread to use for image generation.
        "--background", "3",  # Background to use. 0: Gaussian Noise, 1: Plain white, 2: Quasi crystal, 3: Image.
        "--image_dir", f"{dataset_dir}/#Background_images",  # Image directory to use when background is set to image.
        "--text_color", "#FFFFFF",  # Text's color. "#000000,#FFFFFF" for black to white range.
        # "--space_width", "0",  # Define the width of the spaces between words
        "--name_format", "2",  # Define how the produced files will be named.
        # Rarely needed options.
        # "--font_dir", "data/#Fonts",  # Define a font directory to be used.
        # "--word_split",  # Split on words instead of on characters.
    ]
    print(f"Command: {' '.join(command)}")
    try:
        # Run the command using subprocess.run().
        subprocess.run(command)
    except Exception as error:
        print(f"An error occurred. Error: {error}")


if __name__ == '__main__':
    generate_trdg_images()
