# A file to leverage all other files and train a target PyTorch model.
import torch

import data_setup
import engine
import model_builder
import synthetic_data_gen
import utils


def train(data_dir, device, batch_size, num_epochs, hidden_units, learning_rate):
    # Create DataLoaders with help from data_setup.py
    train_dataloader, test_dataloader = data_setup.create_dataloaders(data_dir, batch_size)

    # Iterate over the training data loader and perform training operations
    # for images, image_texts, bboxes in train_dataloader:
    #     Training code goes here

    # Iterate over the testing data loader and perform testing operations
    # for image, image_text, image_bboxes in test_dataloader:
    #     Testing code goes here


def main() -> None:
    # Setup directories
    data_dir = "training_data/chinese_data/trdg_synthetic_images"
    # Setup target device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Setup hyperparameters
    batch_size = 10
    num_epochs = 5
    hidden_units = 10
    learning_rate = 0.001

    train(data_dir, device, batch_size, num_epochs, hidden_units, learning_rate)


if __name__ == '__main__':
    main()
