import warnings
from collections import Counter

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from sub_ocr.utils import read_image, rescale, pascal_voc_bb, crop_image


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


def visualize_dataset(dataset, num: int = 4, dsp_time: int = 2) -> None:
    """
    Select data at random to show how the preprocessed images in the dataset will be seen by the dataloader.
    :param dataset: Dataset to display.
    :param num: Total number of items to be retrieved from the dataset.
    :param dsp_time: Display time of image. 0 disables timer and keeps the image open until closed.
    """
    print("Visualizing Dataset...")
    ds_len = len(dataset)
    for _ in range(num):
        idx = np.random.randint(ds_len)
        data = dataset[idx]
        print(f"Image Path: {data['image_path']}, Text: {data.get('text')}")
        for key, val in data.items():
            if isinstance(val, np.ndarray) and (len(val.shape) == 2 or len(val.shape) == 3):
                visualize_np_image(val, key, dsp_time)


def visualize_data(image_path: str, labels: list, crop_bbox: bool = True, put_text: bool = False) -> None:
    image, image_height, image_width = read_image(image_path, False)  # Load the image
    if scale := get_scale_factor(image):
        image = rescale(scale, image)

    for label in labels:
        bbox, text = label.get("bbox"), label.get("text")
        if bbox:
            bbox = rescale(scale, bbox=bbox) if scale else bbox
            if crop_bbox and len(labels) == 1:
                _, image = crop_image(image, image_height, image_width, bbox)
            else:
                x_min, y_min, x_max, y_max = map(int, pascal_voc_bb(bbox))
                cv.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Draw the bbox on the image
                if text and put_text:
                    cv.putText(image, text, (x_max, y_max), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    display_image(image, f"Image Rescale Value: {round(scale, 4) if scale else scale}")


def visualize_model(model, image_height: int, image_width: int) -> None:
    """
    Visualize the model with a tensorboard graph.
    """
    with warnings.catch_warnings(action="ignore", category=torch.jit.TracerWarning):
        dummy_input = torch.rand(4, 3, image_height, image_width)
        writer = SummaryWriter(comment="_model_graph")
        writer.add_graph(model, dummy_input)
        writer.close()
        print("\nModel Graph Created! Run 'tensorboard --logdir=runs' to view graph.")


def visualize_char_freq(dataset: list) -> None:
    """
    Visualise the character frequency of the dataset.
    Mean Frequency: This is the average number of occurrences for each character across the dataset. If the mean is
    low, it could suggest that many characters have very few samples.

    Standard Deviation of Frequency: A low standard deviation means the label frequencies are clustered closely
    around the mean, which indicates more balance. A high standard deviation means there is a wide range of
    frequencies, suggesting some labels occur much more frequently than others.

    Balanced Dataset: If both the mean frequency is high (indicating sufficient data per character) and the standard
    deviation is low (indicating a more even distribution), your dataset is likely well-balanced.

    Imbalanced Dataset: A low mean frequency could mean that most labels have very few occurrences, which might lead to
    underfitting for those labels. A high standard deviation indicates that some labels dominate the dataset, which can
    bias the model towards these labels and result in poor generalization to other labels.

    :param dataset: A list from the load_data function in the data_source module.
    """
    print("\nVisualizing Dataset Character Frequency...")
    label_texts = "".join([ds["text"] for labels in dataset for ds in labels[1]])
    counter = Counter(label_texts)
    counts = np.array(list(counter.values()))
    mean_freq, std_freq = np.mean(counts), np.std(counts)

    print(f"Unique characters: {counter}\nUnique characters total: {len(counter):,}\n"
          f"Most common characters: {counter.most_common(10)}\n"
          f"Least common characters: {counter.most_common()[:-11:-1]}\n"
          f"Mean frequency: {mean_freq:,.2f}\nStandard deviation of frequency: {std_freq:,.2f}\n")

    # Plot the distribution of label frequencies
    plt.rcParams["font.family"] = "SimSun"
    plt.figure(figsize=(12, 7)), plt.bar(counter.keys(), counter.values())
    plt.xlabel("Character"), plt.ylabel("Frequency"), plt.title("Character Frequency Distribution")
    plt.tight_layout()
    plt.show()


def visualize_feature_maps(model, input_image: np.ndarray) -> None:
    """
    Visualize model feature maps. Feature maps provide insights into what each convolutional layer is learning.
    """
    print("\nVisualizing model feature maps...")
    input_image = torch.from_numpy(input_image).unsqueeze(0)

    # Extract convolutional layers and their weights
    conv_weights = []  # List to store convolutional layer weights
    conv_layers = []  # List to store convolutional layers
    total_conv_layers = 0  # Counter for total convolutional layers

    def flatten_model_children(model_, level: int = 1) -> None:
        nonlocal total_conv_layers  # Declare to modify outer scope variable
        for name, child in model_.named_children():
            # print(level, name, type(child))
            if isinstance(child, torch.nn.Conv2d):
                conv_weights.append(child.weight), conv_layers.append(child)
                total_conv_layers += 1
            flatten_model_children(child, level + 1)

    flatten_model_children(model)
    print(f"Total convolution layers: {total_conv_layers}")
    # todo: finish implementation of feature maps

    # # Extract feature maps
    # feature_maps = []  # List to store feature maps
    # layer_names = []  # List to store layer names
    # for layer in conv_layers:
    #     print(input_image.shape, layer)
    #     input_image = layer(input_image)
    #     feature_maps.append(input_image)
    #     layer_names.append(str(layer))

    # # Display feature maps shapes
    # print("\nFeature maps shape")
    # for feature_map in feature_maps:
    #     print(feature_map.shape)
    #
    # # Process and visualize feature maps
    # processed_feature_maps = []  # List to store processed feature maps
    # for feature_map in feature_maps:
    #     feature_map = feature_map.squeeze(0)  # Remove the batch dimension
    #     mean_feature_map = torch.sum(feature_map, 0) / feature_map.shape[0]  # Compute mean across channels
    #     processed_feature_maps.append(mean_feature_map.data.cpu().numpy())
