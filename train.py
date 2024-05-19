import logging

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data.build_dataset import TextDetectionDataset, TextRecognitionDataset
from models.detection.db import DB, DBLoss, DBMetrics
from models.recognition.crnn import CRNN, CRNNLoss, CRNNMetrics
from utilities.logger_setup import setup_logging
from utilities.telegram_bot import TelegramBot
from utilities.trainer import ModelTrainer
from utilities.utils import Types, read_chars
from utilities.visualize import visualize_dataset

logger = logging.getLogger(__name__)


def train_text_detection(lang: Types.Language) -> None:
    # Setup hyperparameters
    num_epochs = 10
    batch_size, val_batch_size = 16, 1
    patience, learning_rate = 4, 0.001
    num_workers = 4
    model_name, backbone = Types.db, "deformable_resnet50"
    image_height, image_width = 640, 640

    logger.info(f"Loading {lang} Text Detection Data...")
    train_ds = TextDetectionDataset(lang, Types.train, model_name, image_height, image_width)
    val_ds = TextDetectionDataset(lang, Types.val, model_name, image_height, image_width)
    logger.info(f"Loading Completed... Dataset Size Train: {len(train_ds):,}, Val: {len(val_ds):,}")
    visualize_dataset(train_ds)

    model_params = {"name": model_name, "backbone": backbone, "pretrained": True}
    model = DB(model_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = ReduceLROnPlateau(optimizer, patience=patience)
    train_params = {
        "loss_fn": DBLoss(), "metrics_fn": DBMetrics(), "optimizer": optimizer, "lr_scheduler": lr_scheduler,
        "num_epochs": num_epochs,
        "sanity_check": False,
        "model_dir": "saved models/det models", "model_filename": f"{lang} {model_name} {backbone}"
    }
    trainer = ModelTrainer(model, train_params)
    trainer.set_loaders(train_ds, val_ds, batch_size, val_batch_size, num_workers)
    trainer.load_checkpoint("")
    trainer.train()


def train_text_recognition(lang: Types.Language) -> None:
    # Setup hyperparameters
    num_epochs = 10
    batch_size, val_batch_size = 256, 4096
    patience, learning_rate = 4, 0.0001
    num_workers = 2
    model_name, backbone = Types.crnn, "ctc"
    image_height, image_width = 32, 160

    logger.info(f"Loading {lang} Text Recognition Data...")
    train_ds = TextRecognitionDataset(lang, Types.train, model_name, image_height, image_width)
    val_ds = TextRecognitionDataset(lang, Types.val, model_name, image_height, image_width)
    logger.info(f"Loading Completed... Dataset Size Train: {len(train_ds):,}, Val: {len(val_ds):,}")
    visualize_dataset(train_ds)

    alphabet = read_chars(lang)
    model_params = {"image_height": image_height, "channel_size": 3, "num_class": len(alphabet) + 1}
    model = CRNN(**model_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = ReduceLROnPlateau(optimizer, patience=patience)
    train_params = {
        "loss_fn": CRNNLoss(alphabet), "metrics_fn": CRNNMetrics(alphabet), "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "num_epochs": num_epochs,
        "sanity_check": False,
        "model_dir": "saved models/rec models", "model_filename": f"{lang} {model_name} {backbone}"
    }
    trainer = ModelTrainer(model, train_params)
    trainer.set_loaders(train_ds, val_ds, batch_size, val_batch_size, num_workers)
    trainer.load_checkpoint("")
    trainer.train()


def main(lang: Types.Language, model_type: Types.ModelType) -> None:
    tb = TelegramBot()
    assert model_type in [Types.det, Types.rec], f"Model type must be either {Types.det} or {Types.rec}"
    try:
        if model_type == Types.det:
            train_text_detection(lang)
        else:
            train_text_recognition(lang)
        tb.send_telegram_message(f"Text {model_type} Model Training Done!")
    except Exception as error:
        error_msg = f"During Text {model_type} training an error occurred:\n{error}"
        logger.exception(f"\n{error_msg}")
        tb.send_telegram_message(error_msg)


if __name__ == '__main__':
    setup_logging()
    TelegramBot.credential_file = "credentials/telegram auth.json"
    logger.debug("\n\nLogging Started")
    main(Types.english, Types.rec)
    logger.debug("Logging Ended\n\n")
