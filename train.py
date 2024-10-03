import logging
import os
from pathlib import Path

import torch
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data import build_dataset
from sub_ocr.losses import build_loss
from sub_ocr.metrics import build_metric
from sub_ocr.modeling import build_model
from utilities.logger_setup import setup_logging
from utilities.telegram_bot import TelegramBot
from utilities.trainer import ModelTrainer
from utilities.visualize import visualize_char_freq, visualize_dataset, visualize_model, visualize_feature_maps

logger = logging.getLogger(__name__)


def build_optimizer(model, config: dict):
    opt_params = {"params": model.parameters(), "lr": config["lr"]}
    if "epsilon" in config:
        opt_params.update({"eps": config["epsilon"]})
    if "weight_decay" in config:
        opt_params.update({"weight_decay": config["weight_decay"]})
    optimizer = getattr(torch.optim, config["name"])(**opt_params)
    return optimizer


def build_datasets(lang: str, config: dict) -> tuple:
    dataset = getattr(build_dataset, config["name"])
    return dataset(lang, "train", config), dataset(lang, "val", config)


def display_visuals(model, dataset) -> None:
    visualize_char_freq(dataset.img_data)
    visualize_dataset(dataset, 10, 0)
    visualize_model(model, dataset.image_height, dataset.image_width)
    visualize_feature_maps(model, dataset[0]["image"])


def train_model(model_dir: str, config_name: str, config: dict) -> None:
    params, model = config["Hyperparameters"], build_model(config)
    optimizer = build_optimizer(model, config["Optimizer"])
    loss_fn, metric_fn = build_loss(config["Loss"]), build_metric(config)

    logger.info(f"Loading {config_name} Data...")
    train_ds, val_ds = build_datasets(config["lang"], config["Dataset"])
    logger.info(f"Loading Completed... Dataset Size Train: {len(train_ds):,}, Val: {len(val_ds):,}")
    display_visuals(model, train_ds)

    lr_scheduler = ReduceLROnPlateau(optimizer, patience=params["patience"])
    train_params = {"loss_fn": loss_fn, "metrics_fn": metric_fn, "optimizer": optimizer, "lr_scheduler": lr_scheduler,
                    "num_epochs": params["num_epochs"], "model_dir": model_dir, "model_filename": config_name}
    trainer = ModelTrainer(model, train_params)
    trainer.set_loaders(train_ds, val_ds, params["batch_size"], params["val_batch_size"], params["num_workers"])
    trainer.load_checkpoint("")
    trainer.train()


def main() -> None:
    """
    Setup training from here.
    """
    tb, username = TelegramBot(), os.getlogin()

    model_dir = rf"C:\Users\{username}\OneDrive\Backups\Subtitle OCR Models"
    config_file, lang = Path("configs/rec/PP-OCRv4/ch_PP-OCRv4_rec.yml"), "test"

    config_name = config_file.stem if lang in config_file.stem else f"{lang}_{config_file.stem}"
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    config.update({"lang": lang})
    logger.debug(f"{config_name=}, {config=}")

    try:
        train_model(model_dir, config_name, config)
        tb.send_telegram_message(f"Model Training Completed! {config_name=}")
    except Exception as error:
        error_msg = f"During Model Training an error occurred! {config_name=}\n{error=}"
        logger.exception(f"\n{error_msg}")
        tb.send_telegram_message(error_msg)


if __name__ == '__main__':
    setup_logging("training")
    TelegramBot.credential_file = "credentials/telegram auth.json"
    logger.debug("Logging Started")
    main()
    logger.debug("Logging Ended")
