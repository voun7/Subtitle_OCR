"""
Trains a PyTorch image classification model using device-agnostic code.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.build_dataset import TextDetectionDataset, TextRecognitionDataset
# from models.detection import DB
from models.recognition import CRNN
from utilities.engine import ModelTrainer


def train_text_detection(lang: str) -> None:
    # Setup hyperparameters
    num_epochs = 5
    batch_size = 32
    learning_rate = 0.001
    num_workers = 10

    training_data, val_data = TextDetectionDataset(lang, "train"), TextDetectionDataset(lang, "val")

    # train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # val_dataloader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers)
    #
    # # Create model with help from model_builder.py
    # model = DB()
    # # Set loss function and optimizer
    # loss_fn = nn.CrossEntropyLoss()
    # # Set optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # # Start training with help from engine.py
    # trainer = ModelTrainer(model, loss_fn, optimizer)
    # trainer.set_loaders(train_dataloader, val_dataloader)
    # trainer.train(num_epochs)


def train_text_recognition(lang: str) -> None:
    # Setup hyperparameters
    num_epochs = 5
    batch_size = 32
    learning_rate = 0.001
    num_workers = 10

    training_data, val_data = TextRecognitionDataset(lang, "train"), TextDetectionDataset(lang, "val")

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers)

    # Create model with help from model_builder.py
    model = CRNN()
    # Set loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    # Set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Start training with help from engine.py
    trainer = ModelTrainer(model, loss_fn, optimizer)
    trainer.set_loaders(train_dataloader, val_dataloader)
    trainer.train(num_epochs)


def main() -> None:
    lang = "en"
    train_text_detection(lang)
    # train_text_recognition()


if __name__ == '__main__':
    main()
