from pathlib import Path

import cv2
import torch


def draw_bounding_boxes(image_path: str, bbox_path: str) -> None:
    image = cv2.imread(image_path)  # Load the image
    with open(bbox_path, 'r') as file:  # Read the text file containing bounding box coordinates
        lines = file.readlines()

    for line in lines:  # Iterate over the lines and draw bounding boxes
        line = line.split()
        x1, y1, x2, y2 = map(int, line)  # Change value type from string to int and parse values to variables.
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw the bounding box on the image

    # Display the image with bounding boxes
    cv2.imshow('Image with Bounding Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_model(model: torch.nn.Module, target_dir: str, model_name: str) -> None:
    """
    Saves a PyTorch model to a target directory
    :param model: A target PyTorch model to save.
    :param target_dir: A directory for saving the model to.
    :param model_name: A filename for the saved model. Should include either ".pth" or ".pt" as the file extension.
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)
