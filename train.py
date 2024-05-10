import logging

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data.build_dataset import TextDetectionDataset, TextRecognitionDataset
from models.detection.db import DB, DBLoss, DBMetrics
from models.recognition.crnn import CRNN, CRNNLoss, CRNNMetrics
from utilities.logger_setup import setup_logging
from utilities.telegram_bot import TelegramBot
from utilities.trainer import ModelTrainer
from utilities.utils import Types

# from utilities.visualize import visualize_dataset

logger = logging.getLogger(__name__)


def train_text_detection(lang: Types.Language) -> None:
    # Setup hyperparameters
    num_epochs = 50
    batch_size, val_batch_size = 16, 1
    learning_rate = 0.001
    num_workers = 10
    model_name, backbone = Types.db, "deformable_resnet50"

    logger.info("Loading Text Detection Data...")
    train_ds = TextDetectionDataset(lang, Types.train, model_name)
    val_ds = TextDetectionDataset(lang, Types.val, model_name)
    logger.info(f"Loading Completed... Dataset Size Train: {len(train_ds):,}, Val: {len(val_ds):,}")
    # visualize_dataset(train_ds)

    model_params = {"name": model_name, "backbone": backbone, "pretrained": True}
    model = DB(model_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = ReduceLROnPlateau(optimizer)
    train_params = {
        "loss_fn": DBLoss(), "metrics_fn": DBMetrics(), "optimizer": optimizer, "lr_scheduler": lr_scheduler,
        "num_epochs": num_epochs,
        "sanity_check": False,
        "model_dir": "saved models/det models", "model_filename": f"{model_name} {backbone}"
    }
    trainer = ModelTrainer(model, train_params)
    trainer.set_loaders(train_ds, val_ds, batch_size, val_batch_size, num_workers)
    trainer.load_checkpoint("")
    trainer.train()


def train_text_recognition(lang: Types.Language) -> None:
    # Setup hyperparameters
    num_epochs = 50
    batch_size, val_batch_size = 64, 256
    learning_rate = 0.0001
    num_workers = 10
    model_name = Types.crnn

    logger.info("Loading Text Recognition Data...")
    train_ds = TextRecognitionDataset(lang, Types.train, model_name)
    val_ds = TextRecognitionDataset(lang, Types.val, model_name)
    logger.info(f"Loading Completed... Dataset Size Train: {len(train_ds):,}, Val: {len(val_ds):,}")
    # visualize_dataset(train_ds)

    with open(f"models/recognition/alphabets/{lang}.txt", encoding="utf-8") as file:
        alphabet = "".join([line.rstrip("\n") for line in file])

    model_params = {"image_height": 32, "channel_size": 3, "num_class": len(alphabet), "hidden_size": 256}
    model = CRNN(**model_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = ReduceLROnPlateau(optimizer)
    train_params = {
        "loss_fn": CRNNLoss(alphabet), "metrics_fn": CRNNMetrics(alphabet), "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "num_epochs": num_epochs,
        "sanity_check": False,
        "model_dir": "saved models/rec models", "model_filename": f"{model_name}"
    }
    trainer = ModelTrainer(model, train_params)
    trainer.set_loaders(train_ds, val_ds, batch_size, val_batch_size, num_workers)
    trainer.load_checkpoint("")
    trainer.train()


def main() -> None:
    lang = Types.english
    # tb = TelegramBot()

    # try:
    #     train_text_detection(lang)
    #     tb.send_telegram_message("Text Detection Model Training Done!")
    # except Exception as error:
    #     error_msg = f"During Text Detection training an error occurred:\n{error}"
    #     logger.exception(f"\n{error_msg}")
    #     tb.send_telegram_message(error_msg)

    try:
        train_text_recognition(lang)
        # tb.send_telegram_message("Text Recognition Model Training Done!")
    except Exception as error:
        error_msg = f"During Text Recognition training an error occurred:\n{error}"
        logger.exception(f"\n{error_msg}")
        # tb.send_telegram_message(error_msg)


if __name__ == '__main__':
    setup_logging()
    TelegramBot.credential_file = "credentials/telegram auth.json"
    logger.debug("\n\nLogging Started")
    main()
    logger.debug("Logging Ended\n\n")
