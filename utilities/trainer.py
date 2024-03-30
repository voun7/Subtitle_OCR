import copy
import datetime
import logging
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, model, params: dict) -> None:
        # Here we define the attributes of our class
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        # We start by storing the arguments as attributes to use them later
        self.model = self.init_model(model)
        self.loss_fn = params["loss_fn"]
        self.optimizer = params["optimizer"]
        self.sanity_check = params["sanity_check"]
        self.lr_scheduler = params["lr_scheduler"]
        self.model_file = params["model_file"]

        # These attributes are defined here, but since they are
        # not informed at the moment of creation, we keep them None
        self.train_loader = None
        self.val_loader = None
        self.writer = None

        # These attributes are going to be computed internally
        self.losses = []
        self.val_losses = []
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

    def init_train_dl(self, train_ds, batch_size: int, num_workers: int) -> DataLoader:
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()
        train_dl = DataLoader(train_ds, batch_size, num_workers=num_workers, pin_memory=self.use_cuda)
        return train_dl

    def init_val_dl(self, val_ds, batch_size: int, num_workers: int) -> DataLoader:
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()
        val_dl = DataLoader(val_ds, batch_size, num_workers=num_workers, pin_memory=self.use_cuda)
        return val_dl

    def set_loaders(self, train_ds, val_ds, batch_size: int, val_batch_size: int, num_workers: int) -> None:
        """
        This method allows the user to define which train_loader and val_loader to use.
        """
        self.train_loader = self.init_train_dl(train_ds, batch_size, num_workers)
        self.val_loader = self.init_val_dl(val_ds, val_batch_size, num_workers)

    def set_tensorboard(self, name: str, folder: str = 'runs') -> None:
        """
        This method allows the user to define a SummaryWriter to interface with TensorBoard.
        """
        suffix = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.writer = SummaryWriter(f'{folder}/{name}_{suffix}')

    def _make_train_step_fn(self):
        """
        Builds function that performs a step in the train loop.
        :return: The function that will be called inside the train loop.
        """

        def perform_train_step_fn(x, y):
            # Sets model to TRAIN mode
            self.model.train()
            # Step 1 - Computes our model's predicted output - forward pass
            output = self.model(x)
            # Step 2 - Computes the loss
            loss = self.loss_fn(output, y)
            # Step 3 - Computes gradients for both "a" and "b" parameters
            loss.backward()
            # Step 4 - Updates parameters using gradients and the learning rate
            self.optimizer.step()
            self.optimizer.zero_grad()
            # Returns the loss
            return loss.item()

        return perform_train_step_fn

    def _make_val_step_fn(self):
        """
        Builds function that performs a step in the validation loop.
        """

        def perform_val_step_fn(x, y):
            # Sets model to EVAL mode
            self.model.eval()
            # Step 1 - Computes our model's predicted output - forward pass
            output = self.model(x)
            # Step 2 - Computes the loss
            loss = self.loss_fn(output, y)
            # There is no need to compute Steps 3 and 4, since we don't update parameters during evaluation
            return loss.item()

        return perform_val_step_fn

    def _mini_batch(self, validation: bool = False):
        """
        The mini-batch can be used with both loaders.
        :param validation: Determines which loader and corresponding step function is going to be used
        """
        if validation:
            data_loader, step_fn, mode = self.val_loader, self.val_step_fn, "Validation"
        else:
            data_loader, step_fn, mode = self.train_loader, self.train_step_fn, "Training"

        if data_loader is None:
            return None

        # Once the data loader and step function, this is the same
        # mini-batch loop we had before
        mini_batch_losses, batch_size = [], len(data_loader)
        for index, inputs in enumerate(data_loader):
            x_batch, y_batch = inputs
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            mini_batch_loss = step_fn(x_batch, y_batch)
            mini_batch_losses.append(mini_batch_loss)
            pos = (self.total_epochs + (index + 1) / batch_size)
            print(f"\rEpoch: {pos:.3f}, Batch {mode} Loss: {mini_batch_loss:.6f}", end='', flush=True)

            # break the loop in case of sanity check
            if self.sanity_check is True:
                break

        loss = np.mean(mini_batch_losses)
        return loss

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

    def train(self, n_epochs: int, seed: int = 42) -> None:
        self.set_seed(seed)  # To ensure reproducibility of the training process
        start_time = time.perf_counter()
        # initialize the best loss to a large value
        best_loss, best_model_wts = float('inf'), copy.deepcopy(self.model.state_dict())

        for epoch in range(n_epochs):
            # Keeps track of the numbers of epochs by updating the corresponding attribute
            self.total_epochs += 1

            # Inner loop
            # Performs training using mini-batches
            loss = self._mini_batch()
            self.losses.append(loss)

            # VALIDATION
            with torch.no_grad():  # no gradients in validation!
                # Performs evaluation using mini-batches
                val_loss = self._mini_batch(validation=True)
                self.val_losses.append(val_loss)

            # store best model
            if val_loss < best_loss:
                best_loss, best_model_wts = val_loss, copy.deepcopy(self.model.state_dict())
                self.save_checkpoint()  # store weights into a local file
                logger.info("Saving best model weights!")

            # learning rate schedule
            self.lr_scheduler.step(val_loss)
            if self.lr_scheduler.get_lr() != self.lr_scheduler.get_last_lr():
                logger.info("Loading best model weights!")
                self.model.load_state_dict(best_model_wts)

            logger.info(f"Epoch: {self.total_epochs}/{n_epochs}, Current lr={self.lr_scheduler.get_lr} "
                        f"Training Loss: {loss:.6f}, Validation Loss: {val_loss:.6f}")

            # If a SummaryWriter has been set...
            if self.writer:
                scalars = {'training loss': loss, "validation loss": val_loss}
                # Records both losses for each epoch under the main tag "loss"
                self.writer.add_scalars(main_tag='loss', tag_scalar_dict=scalars, global_step=epoch)

        if self.writer:
            # Closes the writer
            self.writer.close()

        stop_time = time.perf_counter()
        total_time = datetime.timedelta(seconds=stop_time - start_time)
        logger.info(f"Model Training Completed! Total Time: {total_time}")

    def save_checkpoint(self) -> None:
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
        torch.save(checkpoint, self.model_file)

    def load_checkpoint(self) -> None:
        # Loads dictionary
        checkpoint = torch.load(self.model_file)

        # Restore state for model and optimizer
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.total_epochs = checkpoint['epoch']
        self.losses = checkpoint['loss']
        self.val_losses = checkpoint['val_loss']

        self.model.train()  # always use TRAIN for resuming training

    def predict(self, x):
        # Set is to evaluation mode for predictions
        self.model.eval()
        # Takes aNumpy input and make it a float tensor
        x_tensor = torch.as_tensor(x).float()
        # Send input to device and uses model for prediction
        output_tensor = self.model(x_tensor.to(self.device))
        # Set it back to train mode
        self.model.train()
        # Detaches it, brings it to CPU and back to Numpy
        return output_tensor.detach().cpu().numpy()

    def plot_losses(self):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.losses, label='Training Loss', c='b')
        plt.plot(self.val_losses, label='Validation Loss', c='r')
        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        return fig

    def add_graph(self):
        # Fetches a single mini-batch, so we can use add_graph
        if self.train_loader and self.writer:
            x_sample, y_sample = next(iter(self.train_loader))
            self.writer.add_graph(self.model, x_sample.to(self.device))

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
