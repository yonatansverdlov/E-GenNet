"""
Lightning models.
"""
from typing import Callable, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from easydict import EasyDict
from torch.utils.data import DataLoader
from torch_geometric.data import Data
import torchmetrics


# Define the model wrapper class.
# Support training,testing and loading model.


class ModelWrapped(pl.LightningModule):
    """
    The Model wrapper class, support initialization, training, testing.
    """

    def __init__(self, model: nn.Module, config: EasyDict):
        """
        This is the model wrapper for pytorch lightning training.
        Args:
            model: The model.
            config: The config dictionary.
        """
        super(ModelWrapped, self).__init__()
        # The model.
        self.model: nn.Module = model
        # Task.
        self.task = config.task
        # The config.
        self.config = config.type_config.task_specific[self.task]
        # The loss type.
        self.type = config.type
        # The loss.
        self.loss_types = {'k_chain': nn.CrossEntropyLoss(), 'Drugs': nn.MSELoss(),
                           'Kraken': nn.MSELoss(), 'BDE': nn.MSELoss()}
        # The acc.
        self.acc_funcs = {'k_chain': torchmetrics.Accuracy(task="multiclass", num_classes=2),
                          'Drugs': torchmetrics.MeanAbsoluteError(),
                          'Kraken': torchmetrics.MeanAbsoluteError(),
                          'BDE': torchmetrics.MeanAbsoluteError()}
        # The loss criterion.
        self.loss_fun: Callable = self.loss_types[self.type]
        # The accuracy criterion.
        self.acc_fun = self.acc_funcs[self.type]
        # Number of reshuffles.
        self.reshuffle_every_n_epochs = config.type_config.common_to_all_tasks.reshuffle_every_n_epochs

    def configure_optimizers(self) -> Tuple:
        # initialize Adam.
        optimizer = optim.AdamW(
            params=self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.wd,
            eps=1e-07,
            amsgrad=True
        )

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.config.gamma,
                                                                  cooldown=self.config.cooldown)
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "monitor": "train_loss"
        }

        return [optimizer], lr_scheduler_config

    def training_step(self, data_obj: Data, batch_idx: int) -> torch.float:
        """
        Training step.
        Args:
            data_obj: The inputs.
            batch_idx: The data_obj id.

        Returns: The loss on the data_obj.

        """
        # The model.
        model = self.model
        # Move to train mode.
        model.train()
        # The label.
        label = data_obj.label
        # The outs.
        outs = model(data_obj)
        # The loss.
        loss = self.loss_fun(outs, label)  # Compute the loss.
        # The accuracy.
        acc = self.acc_fun(outs, label)  # Compute the Accuracy.
        # Log accuracy, loss.
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True,
                 batch_size=label.size(0), logger=True)  # Update the loss.
        self.log('train_acc', acc, on_step=True, on_epoch=True, sync_dist=True,
                 batch_size=label.size(0), logger=True)  # Update the acc.
        return loss  # Return the loss.

    def test_model(self, data_obj: Data, mode: str) -> torch.float:
        """
        Test model.
        Args:
            data_obj: The batch.
            mode: The mode.

        Returns: The acc

        """
        # The model.
        model = self.model
        # Move to evaluation mode.
        model.eval()
        # The label.
        label = data_obj.label
        # Without grad.
        with torch.no_grad():
            # The output.
            outs = model(data_obj)
            # The loss.
            loss = self.loss_fun(outs, label)
            # The accuracy.
            acc = self.acc_fun(outs, label)
            self.log(f'{mode}_loss', loss, on_step=True, on_epoch=True, sync_dist=True,
                     batch_size=label.size(0), logger=True)  # Update the loss.
            self.log(f'{mode}_acc', acc, on_step=True, on_epoch=True, sync_dist=True,
                     batch_size=label.size(0), logger=True)  # Update the acc.

        return loss  # Return the loss.

    def test_step(self, batch: Data, batch_idx: int) -> torch.float:
        """
        Make the validation step.
        Args:
            batch: The inputs.
            batch_idx: The data_obj id.

        Returns: The task Accuracy on the data_obj.

        """

        return self.test_model(data_obj=batch, mode='test')

    def validation_step(self, batch: Data, batch_idx: int) -> torch.float:
        """
        Make the validation step.
        Args:
            batch: The inputs.
            batch_idx: The data_obj id.

        Returns: The task Accuracy on the data_obj.

        """
        return self.test_model(data_obj=batch, mode='val')

    def compute_metric(self, trainer: pl.Trainer, test_loader: DataLoader, track: str) -> torch.float:
        """
        Computes the accuracy over test loader.
        Args:
            trainer: The trainer object.
            test_loader: The test loader.
            track: What metric to track.loss/val.

        Returns: The accuracy over the test loader.

        """
        return trainer.test(self, dataloaders=test_loader, verbose=False)[0][f'test_{track}_epoch']
