"""
Trains a PyTorch image classification model using device-agnostic code.
"""
import torch

import data_setup
import engine
import model_builder
import utils


def main() -> None:
    # Setup hyperparameters
    num_epochs = 5
    batch_size = 32
    learning_rate = 0.001
    num_classes = 1000  # Assuming text consists of 26 alphabet characters.
    input_height = 100
    input_width = 1900
    hidden_size = 256

    # Setup directories
    data_dir = "training_data/chinese_data/trdg_synthetic_images"

    # Setup target device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create DataLoaders with help from data_setup.py
    train_dataloader, test_dataloader = data_setup.create_dataloaders(data_dir, batch_size)

    # Create model with help from model_builder.py
    model = model_builder.TextRecognitionModel(num_classes, input_height, input_width, hidden_size).to(device)

    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Start training with help from engine.py
    engine.train(model=model, train_dataloader=train_dataloader, test_dataloader=test_dataloader, loss_fn=loss_fn,
                 optimizer=optimizer, epochs=num_epochs, device=device)

    # Save the model with help from utils.py
    utils.save_model(model=model, target_dir="models", model_name="text_rec_model.pth")


if __name__ == '__main__':
    main()
