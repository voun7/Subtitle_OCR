import logging
import random
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, model, params: dict) -> None:
        """
        Model trainer for program.
        The batches from the dataloader should be a dictionary.
        The losses from the loss function should be a dictionary.
        The trainer requires the use of a metrics function that should return a dictionary.
        """
        self.use_cuda = torch.cuda.is_available()
        self.device = "cuda" if self.use_cuda else "cpu"
        self.model = self.init_model(model)
        self.loss_fn, self.metrics_fn, self.optimizer = params["loss_fn"], params["metrics_fn"], params["optimizer"]
        self.lr_scheduler, self.num_epochs = params["lr_scheduler"], params["num_epochs"]
        self.model_dir, self.model_filename = Path(params["model_dir"]), Path(params["model_filename"])
        self.checkpoint_dir = self.model_dir / "Checkpoints"
        self.model_dir.mkdir(parents=True, exist_ok=True), self.checkpoint_dir.mkdir(exist_ok=True)

        # These attributes are defined here, but since they are not needed at the moment of creation, we keep them None
        self.writer = self.train_loader = self.val_loader = None
        # These attributes are going to be computed internally, initialize the best loss to a large value
        self.losses, self.val_losses, self.best_val_loss = {}, {}, float("inf")
        self.metrics, self.val_metrics = {}, {}
        self.total_epochs, self.learning_rates, self.epoch_stop = 0, [], self.num_epochs

    def init_model(self, model):
        if self.use_cuda:
            device_count = torch.cuda.device_count()
            logger.info(f"Using CUDA; {device_count} {'devices' if device_count > 1 else 'device'}.")
            if device_count > 1:
                model = torch.nn.DataParallel(model)
        return model.to(self.device)

    def init_dataloader(self, dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()
        data_loader = DataLoader(
            dataset, batch_size, shuffle, num_workers=num_workers,
            collate_fn=getattr(dataset, "collate_fn", None),
            pin_memory=self.use_cuda if self.use_cuda and batch_size > 2 else False
        )
        logger.debug(f"DataLoader initialized: {vars(data_loader)}")
        return data_loader

    def set_loaders(self, train_ds, val_ds, batch_size: int, val_batch_size: int, num_workers: int) -> None:
        """
        This method allows the user to define which train_loader and val_loader to use.
        """
        self.train_loader = self.init_dataloader(train_ds, batch_size, num_workers, True)
        self.val_loader = self.init_dataloader(val_ds, val_batch_size, num_workers, False)

    def train_step_fn(self, images: torch.Tensor, batch: dict) -> tuple[dict, dict]:
        """
        Function that performs a step in the train loop.
        :param images: Image Tensors
        :param batch: A dictionary with tensor values.
        :return: A loss dict and a metric dict.
        """
        # Step 1 - Computes our model's predicted outputs - forward pass
        outputs = self.model(images)
        # Step 2 - Computes the loss and metrics
        loss = self.loss_fn(outputs, batch)
        metric = self.metrics_fn(outputs, batch, False)
        # Step 3 - Computes gradients for both "x" and "y" parameters
        loss["loss"].backward()
        # Step 4 - Updates parameters using gradients and the learning rate
        self.optimizer.step()
        self.optimizer.zero_grad()
        # Returns the loss and metrics
        return loss, metric

    def val_step_fn(self, images: torch.Tensor, batch: dict) -> tuple[dict, dict]:
        """
        Function that performs a step in the validation loop.
        :param images: Image Tensors
        :param batch: A dictionary with tensor values.
        :return: A loss dict and a metric dict.
        """
        # Step 1 - Computes our model's predicted outputs - forward pass
        outputs = self.model(images)
        # Step 2 - Computes the loss and metrics
        loss = self.loss_fn(outputs, batch)
        metric = self.metrics_fn(outputs, batch, True)
        # There is no need to compute Steps 3 and 4, since we don't update parameters during evaluation
        return loss, metric

    def dict_to_device(self, batch: dict) -> None:
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)

    @staticmethod
    def append_dict_val(dict_1: dict, dict_2: dict) -> None:
        """
        Modify the dictionary's values in place.
        :param dict_1: The dictionary's value that will be used to append dict_2
        :param dict_2: The dictionary's value that will be appended. Should have a list value or no value.
        """
        for key, value in dict_1.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            dict_1[key] = round(value, 5)
            dict_2.setdefault(key, []).append(value)

    def _mini_batch(self, validation: bool = False) -> tuple[dict, dict]:
        """
        The mini-batch can be used with both loaders.
        :param validation: Determines which loader and corresponding step function is going to be used
        """
        if validation:
            data_loader, step_fn, mode = self.val_loader, self.val_step_fn, "Val"
        else:
            data_loader, step_fn, mode = self.train_loader, self.train_step_fn, "Train"

        batch_losses, batch_metrics, num_of_batches, start_time = {}, {}, len(data_loader), perf_counter()
        for index, batch in enumerate(data_loader):
            self.dict_to_device(batch)
            images = batch.pop("image")
            batch_loss, batch_metric = step_fn(images, batch)
            self.append_dict_val(batch_loss, batch_losses), self.append_dict_val(batch_metric, batch_metrics)
            pos = self.total_epochs + (index + 1) / num_of_batches
            print(f"\rEpoch: {pos:.3f}, Batch {mode} Loss: {batch_loss}, Metric: {batch_metric}", end="", flush=True)

        logger.debug(f"Epoch: {self.total_epochs + 1}, Batch {mode} Duration: {self.dur_calc(start_time)}")
        loss = {loss_name: float(np.mean(loss_values)) for loss_name, loss_values in batch_losses.items()}
        metric = {metric_name: float(np.mean(metric_values)) for metric_name, metric_values in batch_metrics.items()}
        return loss, metric

    def set_seed(self, seed: int) -> None:
        """
        Set seed to remove randomness and enable reproducibility of the training process.
        """
        if seed:
            logger.debug(f"Seed set to: {seed}")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            try:
                self.train_loader.sampler.generator.manual_seed(seed)
            except AttributeError:
                pass

    def get_lr(self) -> float:
        """
        Get current learning rate of optimizer.
        """
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    def set_lr(self, new_learning_rate: float) -> None:
        """
        Set new learning rate for optimizer.
        """
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_learning_rate

    @staticmethod
    def clear_previous_print() -> None:
        """
        Clear the previous print line. Should be used before prints or logs in the training loop.
        """
        print(end="\r", flush=True)

    @staticmethod
    def dur_calc(start_time: float) -> timedelta:
        """
        Duration calculator.
        :return: The duration
        """
        return timedelta(seconds=round(perf_counter() - start_time))

    def update_writer(self) -> None:
        """
        Update the writer with the checkpoint data when the total epochs start above zero.
        """
        if not self.total_epochs:
            return  # no checkpoint or no recorded data in checkpoint
        for i in range(self.total_epochs):
            self.writer.add_scalar("Learning Rate", self.learning_rates[i], i + 1)
            self.writer.add_scalars("Training Loss", {k: self.losses[k][i] for k in self.losses}, i + 1)
            self.writer.add_scalars("Training Metric", {k: self.metrics[k][i] for k in self.metrics}, i + 1)
            self.writer.add_scalars("Validation Loss", {k: self.val_losses[k][i] for k in self.val_losses}, i + 1)
            self.writer.add_scalars("Validation Metric", {k: self.val_metrics[k][i] for k in self.val_metrics}, i + 1)
        logger.info("Writer Updated with Checkpoint data.")

    def train_model(self, seed: int = None) -> None:
        assert self.train_loader and self.val_loader, "Train or Val data loader has not been set!"
        start_time, self.writer = perf_counter(), SummaryWriter()
        best_model_wts = deepcopy(self.model.state_dict())  # Initial copy of model weights is saved
        self.set_seed(seed), self.update_writer()

        for _ in range(self.epoch_stop):
            current_lr = self.get_lr()
            self.model.train()  # Sets model to TRAIN mode
            loss, metric = self._mini_batch()  # Performs training using mini-batches
            self.model.eval()  # Sets model to EVAL mode
            with torch.no_grad():  # no gradients in validation!
                val_loss, val_metric = self._mini_batch(validation=True)  # Performs validation using mini-batches

            self.total_epochs += 1  # Keeps track of the total numbers of epochs
            self.record_values(loss, metric, val_loss, val_metric)
            # store best model
            if val_loss["loss"] < self.best_val_loss:
                self.best_val_loss, best_model_wts = val_loss["loss"], deepcopy(self.model.state_dict())
                self.clear_previous_print()
                logger.info("Saving best model weights!")
                self.save_checkpoint()
            # learning rate schedule
            self.lr_scheduler.step(val_loss["loss"])
            if current_lr != self.lr_scheduler.get_last_lr()[0]:
                self.clear_previous_print()
                logger.info("Learning rate changed. Loading best model weights!")
                self.model.load_state_dict(best_model_wts)

        self.writer.close()  # Closes the writer
        logger.info(f"Model Training Completed. Duration: {self.dur_calc(start_time)}")
        logger.debug(f"Trainer Values:\n{self.losses=}\n{self.val_losses=}\n{self.metrics=}\n{self.val_metrics=}\n"
                     f"{self.learning_rates=}")

    def record_values(self, loss: dict, metric: dict, val_loss: dict, val_metric: dict) -> None:
        """
        The logger and tensorboard writer will be used to record the values from the training and validation loop.
        """
        self.append_dict_val(loss, self.losses), self.append_dict_val(metric, self.metrics)
        self.append_dict_val(val_loss, self.val_losses), self.append_dict_val(val_metric, self.val_metrics)
        current_lr = self.get_lr()
        self.learning_rates.append(current_lr)
        if hasattr(self.metrics_fn, "get_metric"):
            more_val_metric = self.metrics_fn.get_metric()
            self.append_dict_val(more_val_metric, self.val_metrics), val_metric.update(more_val_metric)
        self.clear_previous_print()
        logger.info(f"Epoch: {self.total_epochs}/{self.num_epochs}, Current lr={current_lr},\n"
                    f"Train Loss: {loss}, Metric: {metric}{'\n' if len(loss) > 1 else ' | '}"
                    f"Val Loss: {val_loss}, Metric: {val_metric}")

        # Records the values for each epoch on the writer
        self.writer.add_scalar("Learning Rate", current_lr, self.total_epochs)
        self.writer.add_scalars("Training Loss", loss, self.total_epochs)
        self.writer.add_scalars("Training Metric", metric, self.total_epochs)
        self.writer.add_scalars("Validation Loss", val_loss, self.total_epochs)
        self.writer.add_scalars("Validation Metric", val_metric, self.total_epochs)

    def save_checkpoint(self) -> None:
        """
        Build and save a checkpoint with a dictionary containing all the objects for resuming training.
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(), "optimizer_state_dict": self.optimizer.state_dict(),
            "total_epochs": self.total_epochs, "learning_rates": self.learning_rates,
            "losses": self.losses, "val_losses": self.val_losses, "best_val_loss": self.best_val_loss,
            "metrics": self.metrics, "val_metrics": self.val_metrics
        }
        torch.save(checkpoint, self.checkpoint_dir / f"{self.model_filename} (checkpoint) ({self.best_val_loss}).pt")

    def load_checkpoint(self, checkpoint_file: str, new_learning_rate: float = None,
                        set_best_val_loss: bool = True) -> None:
        """
        Load model state, optimizer state and other values from checkpoint file to resuming training model.
        :param checkpoint_file: Name or location of checkpoint file.
        :param new_learning_rate: New learning rate to override checkpoints previous learning rate.
        :param set_best_val_loss: If True the best val loss from the checkpoint file will be used.
        """
        if not (checkpoint_file.endswith(".pt") and Path(self.checkpoint_dir, checkpoint_file).exists()):
            logger.warning("Checkpoint File Not Loaded or Does Not Exist.")
            return
        checkpoint_file = self.checkpoint_dir / checkpoint_file
        logger.info(f"\nCheckpoint File Loaded! File: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, weights_only=False)  # Load checkpoint dict
        # Restore state for model and optimizer
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if new_learning_rate:
            self.set_lr(new_learning_rate)
        self.total_epochs, self.learning_rates = checkpoint["total_epochs"], checkpoint["learning_rates"]
        self.losses, self.val_losses = checkpoint["losses"], checkpoint["val_losses"]
        if set_best_val_loss:
            self.best_val_loss = checkpoint["best_val_loss"]
        self.metrics, self.val_metrics = checkpoint["metrics"], checkpoint["val_metrics"]
        self.num_epochs += self.total_epochs  # update the overall number of epochs
        logger.info(f"Model Checkpoint Loaded: Model Params No: {len(list(self.model.named_parameters()))},\n"
                    f"Optimizer: {self.optimizer},\nTotal Epochs: {self.total_epochs}, "
                    f"Best Validation Loss: {self.best_val_loss}\n"
                    f"Loss Keys: {list(self.losses)}\nVal Loss Keys: {list(self.val_losses)}\n"
                    f"Metric Keys: {list(self.metrics)}\nVal Metric Keys: {list(self.val_metrics)}")
        logger.debug(f"Checkpoint Values:\n{self.losses=}\n{self.val_losses=}\n{self.metrics=}\n{self.val_metrics=}\n"
                     f"{self.learning_rates=}")

    def save_model(self) -> None:
        """
        Save a traced model with value of the last validation loss.
        """
        last_val_loss = round(self.val_losses["loss"][-1], 5) if self.val_losses.get("loss") else None
        save_path = self.model_dir / f"{self.model_filename} ({last_val_loss}).pt"
        self.model.eval()
        with torch.inference_mode():
            dummy_input = torch.rand(1, 3, 64, 320, device=self.device)
            traced_model = torch.jit.trace(self.model, dummy_input)
            torch.jit.save(traced_model, save_path)
        logger.info(f"Model Saved! Path: {save_path}")

    def create_model_checkpoint(self, model_file: str) -> None:
        """
        Use an already saved model to create a checkpoint file that can be used for fine-tuning.
        :param model_file: Path to model file.
        """
        assert Path(model_file).exists(), "File does not exist!"
        self.model.load_state_dict(torch.load(model_file, self.device, weights_only=True))
        self.save_checkpoint()
        logger.info("Model Checkpoint created!")


def plot_checkpoint(model_checkpoint_file: str) -> None:
    """
    Use data from model checkpoint to plot the data recorded during training.
    command to view graph: tensorboard --logdir=runs
    """
    writer, checkpoint = SummaryWriter(comment="checkpoint_plt"), torch.load(model_checkpoint_file)
    total_epochs, learning_rates = checkpoint["total_epochs"], checkpoint["learning_rates"]
    losses, val_losses, best_val_loss = checkpoint["losses"], checkpoint["val_losses"], checkpoint["best_val_loss"]
    metrics, val_metrics = checkpoint["metrics"], checkpoint["val_metrics"]
    losses.update(metrics), val_losses.update(val_metrics)
    train_keys, val_keys = list(losses), list(val_losses)
    logger.info(f"Plotting Loaded Model Checkpoint: \tTotal Epochs: {total_epochs}, "
                f"Best Validation Loss: {best_val_loss}\nTraining Keys: {train_keys},\nValidation Keys: {val_keys}")

    for idx in range(total_epochs):
        writer.add_scalar("Learning Rate", learning_rates[idx], idx)
        writer.add_scalars("Training Values", {k: losses[k][idx] for k in train_keys}, idx)
        writer.add_scalars("Validation Values", {k: val_losses[k][idx] for k in val_keys}, idx)
