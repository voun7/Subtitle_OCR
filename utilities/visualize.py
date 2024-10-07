import logging
import warnings
from collections import Counter

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from sub_ocr.utils import read_image, rescale, pascal_voc_bb, crop_image

logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)


def get_scale_factor(img_frame: np.ndarray, img_target_height: int = 600) -> float:
    """
    Returns a value that can be used to resize the image to be equal or less than the target height.
    """
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


def visualize_dataset(dataset, num: int, dsp_time: int) -> None:
    """
    Select data at random from the dataset to show how the preprocessed images will be seen by the dataloader.
    :param dataset: Dataset to display.
    :param num: Total number of items to be retrieved from the dataset.
    :param dsp_time: Display time of image. 0 disables timer and keeps the image open until closed.
    """
    print("\nVisualizing Dataset...")
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
          f"Mean frequency: {mean_freq:,.2f}\nStandard deviation of frequency: {std_freq:,.2f}")

    # Plot the distribution of label frequencies
    plt.rcParams["font.family"] = "SimSun"
    plt.figure(figsize=(12, 7)), plt.bar(counter.keys(), counter.values())
    plt.xlabel("Character"), plt.ylabel("Frequency"), plt.title("Character Frequency Distribution")
    plt.tight_layout()
    plt.show()


def visualize_model(model, input_image: np.ndarray) -> None:
    """
    Visualize the model with a tensorboard graph.
    """
    with warnings.catch_warnings(action="ignore", category=torch.jit.TracerWarning):
        input_image = torch.from_numpy(input_image).unsqueeze(0)
        writer = SummaryWriter(comment="_model_graph")
        writer.add_graph(model, input_image)
        writer.close()
        print("\nModel Graph Created! Run 'tensorboard --logdir=runs' to view graph.")


def visualize_feature_maps(model, input_image: np.ndarray, debug: bool = False) -> None:
    """
    Visualize model feature maps. Feature maps provide insights into what each 2D convolutional layer is learning.
    To capture the feature maps, forward hooks are registered to the conv layers.
    A forward hook captures the output of a layer after the forward pass.
    All feature maps will be displayed when debug is True.
    """
    print("\nVisualizing Model Feature Maps...")
    input_image = torch.from_numpy(input_image).unsqueeze(0)
    feature_maps = {}  # A dictionary to store the feature maps

    # Hook function to store the outputs
    def hook_fn(module, input_, output) -> None:
        if debug:
            print(f"Feature map layer: {module},\nInput shape: {input_[0].shape}, Output shape: {output.shape}\n")
        feature_maps[module] = output

    # Automatically find all Conv2d layers and register hooks
    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            hook = layer.register_forward_hook(hook_fn)
            hooks.append(hook)
            if debug:
                print(f"Hook registered to Name: {name}, Layer: {layer}")
    print(f"Total Convolution 2D Layers: {len(hooks)}")

    _ = model(input_image)  # Extract feature maps with a forward pass

    # Remove the hooks to prevent memory leaks
    for hook in hooks:
        hook.remove()

    # Visualize feature maps from all layers
    for layer_num, (layer, feature_map) in enumerate(feature_maps.items(), 1):
        if debug:
            print(f"Visualizing feature maps #{layer_num} from layer {layer}")
        else:
            if layer_num == 4:
                break
        num_feature_maps = feature_map.size(1)  # Number of feature maps (channels)
        num_cols = min(6, num_feature_maps)  # Number of columns for the plot grid
        num_rows = (num_feature_maps + num_cols - 1) // num_cols  # Calculate number of rows

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
        axes = axes.flatten()  # Flatten axes for easy indexing

        # Visualize all feature maps in the current layer
        for i in range(num_feature_maps):
            axes[i].imshow(feature_map[0, i].detach().cpu())
            axes[i].axis('off')

        # Remove any unused subplots (if feature maps < num_cols*num_rows)
        for i in range(num_feature_maps, len(axes)):
            fig.delaxes(axes[i])

        fig.suptitle(f"Feature maps #{layer_num}, Layer: {layer}")  # Set the title of the figure
        plt.tight_layout()
        plt.show()
