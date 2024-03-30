"""
Trains a PyTorch image classification model using device-agnostic code.
"""
import logging

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import v2

from data.build_dataset import TextDetectionDataset, TextRecognitionDataset
from models.detection import DB
from models.recognition import CRNN
from utilities.logger_setup import setup_logging
from utilities.telegram_bot import TelegramBot
from utilities.trainer import ModelTrainer

logger = logging.getLogger(__name__)


def train_text_detection(lang: str) -> None:
    # Setup hyperparameters
    num_epochs = 100
    batch_size = 32
    val_batch_size = 64
    learning_rate = 0.001
    num_workers = 10

    train_transformer = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomRotation(45),
        v2.RandomResizedCrop(96, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
    val_transformer = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    train_ds = TextDetectionDataset(lang, "train", train_transformer)
    val_ds = TextDetectionDataset(lang, "val", val_transformer)

    model_params = {
        "input_shape": (3, 96, 96),
        "initial_filters": 8,
        "num_fc1": 100,
        "dropout_rate": 0.25,
        "num_classes": 2,
    }
    model = DB(model_params)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = ReduceLROnPlateau(optimizer)
    train_params = {
        "loss_fn": loss_fn,
        "optimizer": optimizer,
        "sanity_check": True,
        "lr_scheduler": lr_scheduler,
        "model_file": "./saved models/detection_model.pt",
    }
    trainer = ModelTrainer(model, train_params)
    trainer.set_loaders(train_ds, val_ds, batch_size, val_batch_size, num_workers)
    trainer.train(num_epochs)


def train_text_recognition(lang: str) -> None:
    # Setup hyperparameters
    num_epochs = 100
    batch_size = 32
    val_batch_size = 64
    learning_rate = 0.001
    num_workers = 10

    train_transformer = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomRotation(45),
        v2.RandomResizedCrop(96, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
    val_transformer = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    train_ds = TextRecognitionDataset(lang, "train", train_transformer)
    val_ds = TextRecognitionDataset(lang, "val", val_transformer)

    model_params = {
        "input_shape": (3, 96, 96),
        "initial_filters": 8,
        "num_fc1": 100,
        "dropout_rate": 0.25,
        "num_classes": 2,
    }
    model = CRNN(model_params)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = ReduceLROnPlateau(optimizer)
    train_params = {
        "loss_fn": loss_fn,
        "optimizer": optimizer,
        "sanity_check": True,
        "lr_scheduler": lr_scheduler,
        "model_file": "./saved models/recognition_model.pt",
    }
    trainer = ModelTrainer(model, train_params)
    trainer.set_loaders(train_ds, val_ds, batch_size, val_batch_size, num_workers)
    trainer.train(num_epochs)


def main() -> None:
    lang = "en"
    tb = TelegramBot()
    train_text_detection(lang)
    tb.send_telegram_message("Text Detection Model Training Done!")
    train_text_recognition(lang)
    tb.send_telegram_message("Text Recognition Model Training Done!")


if __name__ == '__main__':
    setup_logging()
    TelegramBot.credential_file = "credentials/telegram auth.json"
    logger.debug("\n\nLogging Started")
    main()
    logger.debug("Logging Ended\n\n")
