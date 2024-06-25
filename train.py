import logging
import os

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data.build_dataset import TextDetectionDataset, TextRecognitionDataset
from sub_ocr.models.detection import DB, DBLoss, DBMetrics
from sub_ocr.models.recognition import SVTR, CTCLoss, RecMetrics
from sub_ocr.utils import Types, read_chars
from utilities.logger_setup import setup_logging
from utilities.telegram_bot import TelegramBot
from utilities.trainer import ModelTrainer
from utilities.visualize import visualize_dataset

logger = logging.getLogger(__name__)


def train_text_detection(lang: Types.Language, model_dir: str) -> None:
    # Setup hyperparameters
    num_epochs = 10
    batch_size, val_batch_size = 8, 1
    patience, learning_rate = 2, 0.001
    num_workers = 4
    model_name, backbone = Types.db, "deformable_resnet50"
    image_h, image_w = 640, 640

    logger.info(f"Loading {lang} Text Detection Data...")
    train_ds = TextDetectionDataset(lang, Types.train, model_name, image_h, image_w)
    val_ds = TextDetectionDataset(lang, Types.val, model_name, image_h, image_w)
    logger.info(f"Loading Completed... Dataset Size Train: {len(train_ds):,}, Val: {len(val_ds):,}")
    visualize_dataset(train_ds)

    model = DB(**{"backbone_name": backbone, "pretrained": False})
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = ReduceLROnPlateau(optimizer, patience=patience)
    train_params = {
        "loss_fn": DBLoss(), "metrics_fn": DBMetrics(), "optimizer": optimizer, "lr_scheduler": lr_scheduler,
        "num_epochs": num_epochs,
        "sanity_check": False,
        "model_dir": model_dir, "model_filename": f"{lang} {model_name} {backbone}"
    }
    trainer = ModelTrainer(model, train_params)
    trainer.set_loaders(train_ds, val_ds, batch_size, val_batch_size, num_workers)
    trainer.load_checkpoint("")
    trainer.train()


def train_text_recognition(lang: Types.Language, model_dir: str) -> None:
    # Setup hyperparameters
    num_epochs = 10
    batch_size, val_batch_size = 32, 32
    patience, learning_rate = 2, 0.0001
    num_workers = 2
    model_name, backbone = Types.svtr, "svtr_base"
    image_h, image_w = 32, 640

    logger.info(f"Loading {lang} Text Recognition Data...")
    train_ds = TextRecognitionDataset(lang, Types.train, model_name, image_h, image_w)
    val_ds = TextRecognitionDataset(lang, Types.val, model_name, image_h, image_w)
    logger.info(f"Loading Completed... Dataset Size Train: {len(train_ds):,}, Val: {len(val_ds):,}")
    visualize_dataset(train_ds)

    alphabet = read_chars(lang)
    model = SVTR(**{"backbone_name": backbone, "img_size": (image_h, image_w), "num_class": len(alphabet) + 1,
                    "max_text_len": 80})
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = ReduceLROnPlateau(optimizer, patience=patience)
    train_params = {
        "loss_fn": CTCLoss(alphabet), "metrics_fn": RecMetrics(alphabet), "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "num_epochs": num_epochs,
        "sanity_check": False,
        "model_dir": model_dir, "model_filename": f"{lang} {model_name} {backbone}"
    }
    trainer = ModelTrainer(model, train_params)
    trainer.set_loaders(train_ds, val_ds, batch_size, val_batch_size, num_workers)
    trainer.load_checkpoint("")
    trainer.train()


def main(lang: Types.Language, model_type: Types.ModelType) -> None:
    tb, username = TelegramBot(), os.getlogin()
    model_dir = rf"C:\Users\{username}\OneDrive\Backups\Subtitle OCR Models"
    assert model_type in [Types.det, Types.rec], f"Model type must be either {Types.det} or {Types.rec}"
    try:
        if model_type == Types.det:
            train_text_detection(lang, model_dir)
        else:
            train_text_recognition(lang, model_dir)
        tb.send_telegram_message(f"Text {model_type} Model Training Done!")
    except Exception as error:
        error_msg = f"During Text {model_type} training an error occurred:\n{error}"
        logger.exception(f"\n{error_msg}")
        tb.send_telegram_message(error_msg)


if __name__ == '__main__':
    setup_logging("training")
    TelegramBot.credential_file = "credentials/telegram auth.json"
    logger.debug("Logging Started")
    main(Types.english, Types.rec)
    logger.debug("Logging Ended")
