import subprocess


def generate_trdg_images():
    command = [
        "trdg",
        "--output_dir", "data/trdg_synthetic_images",  # The output directory.
        "--language", "cn",  # The language to use.
        "--count", "1000",  # The number of images to be created.
        "--random_sequences",  # Use random sequences as the source text for the generation.
        "--length", "20",  # Define how many words should be included in each generated sample.
        "--random",  # Define if the produced string will have variable word count (with --length being the maximum).
        "--format", "150",  # Define the height of the produced images if horizontal, else the width.
        "--thread_count", "6",  # Define the number of thread to use for image generation.
        "--background", "3",  # Background to use. 0: Gaussian Noise, 1: Plain white, 2: Quasi crystal, 3: Image.
        "--image_dir", "data/#Background_images",  # Define an image directory to use when background is set to image.
        "--output_bboxes", "1",  # Define if the generator will return bounding boxes for the text.
        "--text_color", "#FFFFFF",  # Text's color. "#000000,#FFFFFF" for black to white range.
        "--space_width", "0",  # Define the width of the spaces between words
        # "--character_spacing", "0",  # Define the width of the spaces between characters. 2 means two pixels.
        "--margins", "15,15,15,15",  # Define the margins around the text when rendered. In pixels.
        "--fit",  # Apply a tight crop around the rendered text.
        # "--use_wikipedia",  # Use Wikipedia as the source text for the generation.
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
