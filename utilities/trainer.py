import logging
import random
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, model, params: dict) -> None:
        """
        Model trainer for program.
        The batches from the dataloader should be a dictionary.
        The losses from the loss function should be a dictionary.
        The trainer supports optional use of metrics that should be a dictionary.
        """
        # Here we define the attributes of our class
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        # We start by storing the arguments as attributes to use them later
        self.model = self.init_model(model)
        self.loss_fn = params["loss_fn"]
        self.metrics = params.get("metrics")
        self.optimizer = params["optimizer"]
        self.sanity_check = params["sanity_check"]
        self.lr_scheduler = params["lr_scheduler"]
        self.model_dir = Path(params["model_dir"])
        self.model_filename = Path(params["model_filename"])
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # These attributes are defined here, but since they are not needed at the moment of creation, we keep them None
        self.train_loader = None
        self.val_loader = None

        self.writer = SummaryWriter()

        # These attributes are going to be computed internally
        self.losses = {}
        self.val_losses = {}
        self.total_epochs = 0

        # Creates the train_step function for our model, loss function and optimizer
        # Note: there are NO ARGS there! It makes use of the class attributes directly
        self.train_step_fn = self._make_train_step_fn()
        # Creates the val_step function for our model and loss
        self.val_step_fn = self._make_val_step_fn()

    def init_model(self, model):
        if self.use_cuda:
            device_count = torch.cuda.device_count()
            logger.info(f"Using CUDA; {device_count} {'devices' if device_count > 1 else 'device'}.")
            if device_count > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        return model

    def init_dataloader(self, dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()
        data_loader = DataLoader(dataset, batch_size, shuffle, num_workers=num_workers,
                                 collate_fn=getattr(dataset, "collate_fn", None), pin_memory=self.use_cuda)
        return data_loader

    def set_loaders(self, train_ds, val_ds, batch_size: int, val_batch_size: int, num_workers: int) -> None:
        """
        This method allows the user to define which train_loader and val_loader to use.
        """
        self.train_loader = self.init_dataloader(train_ds, batch_size, num_workers, True)
        self.val_loader = self.init_dataloader(val_ds, val_batch_size, num_workers, False)

    def _make_train_step_fn(self):
        """
        Builds function that performs a step in the train loop.
        :return: The function that will be called inside the train loop.
        """

        def perform_train_step_fn(x: torch.Tensor, y: dict) -> tuple[dict, dict | None]:
            """
            :param x: Image Tensor
            :param y: A dictionary with tensor values.
            :return: A loss dict and a metric dict.
            """
            # Sets model to TRAIN mode
            self.model.train()
            # Step 1 - Computes our model's predicted output - forward pass
            output = self.model(x)
            # Step 2 - Computes the loss and metrics
            loss = self.loss_fn(output, y)
            metric = self.compute_metrics(output, y)
            # Step 3 - Computes gradients for both "a" and "b" parameters
            loss['loss'].backward()
            # Step 4 - Updates parameters using gradients and the learning rate
            self.optimizer.step()
            self.optimizer.zero_grad()
            # Returns the loss
            return loss, metric

        return perform_train_step_fn

    def _make_val_step_fn(self):
        """
        Builds function that performs a step in the validation loop.
        """

        def perform_val_step_fn(x: torch.Tensor, y: dict) -> tuple[dict, dict | None]:
            """
            :param x: Image Tensor
            :param y: A dictionary with tensor values.
            :return: A loss dict and a metric dict.
            """
            # Sets model to EVAL mode
            self.model.eval()
            # Step 1 - Computes our model's predicted output - forward pass
            output = self.model(x)
            # Step 2 - Computes the loss and metrics
            loss = self.loss_fn(output, y)
            metric = self.compute_metrics(output, y, True)
            # There is no need to compute Steps 3 and 4, since we don't update parameters during evaluation
            return loss, metric

        return perform_val_step_fn

    def dict_to_device(self, batch: dict) -> dict:
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
        return batch

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
            value = round(value, 4)
            dict_1[key] = value
            dict_2.setdefault(key, []).append(value)

    def _mini_batch(self, validation: bool = False) -> tuple[dict, dict | None]:
        """
        The mini-batch can be used with both loaders.
        :param validation: Determines which loader and corresponding step function is going to be used
        """
        if validation:
            data_loader, step_fn, mode = self.val_loader, self.val_step_fn, "Validation"
        else:
            data_loader, step_fn, mode = self.train_loader, self.train_step_fn, "Training"

        mini_batch_losses, mini_batch_metrics, metric, batch_size = {}, {}, None, len(data_loader)
        for index, batch in enumerate(data_loader):
            batch = self.dict_to_device(batch)
            mini_batch_loss, mini_batch_metric = step_fn(batch["image"], batch)
            self.append_dict_val(mini_batch_loss, mini_batch_losses)
            if self.metrics:
                self.append_dict_val(mini_batch_metric, mini_batch_metrics)
            pos = (self.total_epochs + (index + 1) / batch_size)
            metric_txt = f" Metric: {mini_batch_metric}," if self.metrics else ""
            print(f"\rEpoch: {pos:.3f}, Batch {mode}{metric_txt} Loss: {mini_batch_loss}", end='', flush=True)

            # break the loop in case of sanity check
            if self.sanity_check is True:
                break

        loss = {loss_name: np.mean(loss_values) for loss_name, loss_values in mini_batch_losses.items()}
        if self.metrics:
            metric = {metric_name: np.mean(metric_values) for metric_name, metric_values in mini_batch_metrics.items()}
        return loss, metric

    def compute_metrics(self, predictions: torch.Tensor, batch: dict, validation: bool = False) -> dict | None:
        if self.metrics:
            return self.metrics(predictions, batch, validation)

    def set_seed(self, seed: int) -> None:
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
        get current learning rate
        """
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    @staticmethod
    def clear_previous_print() -> None:
        """
        Clear the previous print line.
        """
        print(end="\r", flush=True)

    def train(self, n_epochs: int, seed: int = 42) -> None:
        assert self.train_loader and self.val_loader
        self.set_seed(seed)  # To ensure reproducibility of the training process
        start_time = perf_counter()
        # initialize the best loss to a large value
        best_loss, best_model_wts = float('inf'), deepcopy(self.model.state_dict())

        for epoch in range(n_epochs):
            current_lr = self.get_lr()
            # Inner Loops
            # Performs training using mini-batches
            loss, metric = self._mini_batch()
            self.append_dict_val(loss, self.losses)
            # Validation
            with torch.no_grad():  # no gradients in validation!
                # Performs evaluation using mini-batches
                val_loss, val_metric = self._mini_batch(validation=True)
                self.append_dict_val(val_loss, self.val_losses)

            # store best model
            if val_loss["loss"] < best_loss:
                best_loss, best_model_wts = val_loss["loss"], deepcopy(self.model.state_dict())
                self.save_checkpoint(best_loss)  # store weights into a local file
                self.clear_previous_print()
                logger.info("Saving best model weights!")

            # learning rate schedule
            self.lr_scheduler.step(val_loss["loss"])
            if current_lr != self.lr_scheduler.get_last_lr()[0]:
                self.clear_previous_print()
                logger.info("Loading best model weights!")
                self.model.load_state_dict(best_model_wts)

            self.total_epochs += 1  # Keeps track of the total numbers of epochs
            self.clear_previous_print()
            if hasattr(self.metrics, "gather_val_metrics"):
                val_metric.update(self.metrics.gather_val_metrics())
            metric_txt = f"Training Metric: {metric}, Validation Metric: {val_metric}, \n" if self.metrics else ""
            logger.info(f"Epoch: {self.total_epochs}/{n_epochs}, Current lr={current_lr}, \n{metric_txt}"
                        f"Training Loss: {loss}, Validation Loss: {val_loss}\n")

            # Records the values for each epoch
            self.writer.add_scalar("learning rate", current_lr, epoch)
            self.writer.add_scalars("training loss", loss, epoch)
            self.writer.add_scalars("validation loss", val_loss, epoch)
            if self.metrics:
                # For validation metrics that are generated per epoch instead of per batch
                self.writer.add_scalars("training metric", metric, epoch)
                self.writer.add_scalars("validation metric", val_metric, epoch)

        self.writer.close()  # Closes the writer

        self.save_model()
        total_time = timedelta(seconds=round(perf_counter() - start_time))
        logger.info(f"Model Training Completed & Model Saved! Total Time: {total_time}")
        logger.debug(f"{self.losses = }, \n{self.val_losses = }")

    def save_checkpoint(self, best_loss) -> None:
        """
        Builds dictionary with all elements for resuming training.
        """
        checkpoint = {
            'epoch': self.total_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.losses,
            'val_loss': self.val_losses
        }
        torch.save(checkpoint, self.model_dir.joinpath(f"{self.model_filename} (checkpoint) ({best_loss}).pt"))

    def load_checkpoint(self, model_filename: str) -> None:
        # Loads dictionary
        checkpoint = torch.load(model_filename)
        # Restore state for model and optimizer
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.total_epochs = checkpoint['epoch']
        self.losses = checkpoint['loss']
        self.val_losses = checkpoint['val_loss']
        logger.info(f"Loading checkpoint: model: {self.model}, optimizer: {self.optimizer}, "
                    f"total epochs: {self.total_epochs}, \nlosses: {self.losses}, \nval losses: {self.val_losses}")
        self.model.train()  # always use TRAIN for resuming training

    def save_model(self) -> None:
        torch.save(self.model.state_dict(), self.model_dir.joinpath(f"{self.model_filename}.pt"))

    def add_graph(self) -> None:
        # Fetches a single mini-batch, so we can use add_graph
        if self.train_loader and self.writer:
            x_sample, y_sample = next(iter(self.train_loader))
            self.writer.add_graph(self.model, x_sample.to(self.device))
