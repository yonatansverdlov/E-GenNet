import lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from easydict import EasyDict
from torch.utils.data import DataLoader
from torch_geometric.data import Data

class ModelWrapped(pl.LightningModule):
    """
    PyTorch Lightning model wrapper for training and testing.
    """
    def __init__(self, model: nn.Module, config: EasyDict):
        super().__init__()
        self.model = model
        self.task = config.task
        self.config = config.type_config.task_specific[self.task]
        self.type = config.type
        
        loss_map = {'k_chain': nn.CrossEntropyLoss(), 'Drugs': nn.MSELoss(), 'Kraken': nn.MSELoss(),
                    'BDE': nn.MSELoss(), 'Hard': nn.CrossEntropyLoss()}
        metric_map = {'k_chain': torchmetrics.Accuracy(task="multiclass", num_classes=2),
                      'Drugs': torchmetrics.MeanAbsoluteError(),
                      'Kraken': torchmetrics.MeanAbsoluteError(),
                      'BDE': torchmetrics.MeanAbsoluteError(),
                      'Hard': torchmetrics.Accuracy(task="multiclass", num_classes=2)}
        
        self.loss_fun = loss_map[self.type]
        self.acc_fun = metric_map[self.type]
        self.reshuffle_every_n_epochs = config.type_config.common_to_all_tasks.reshuffle_every_n_epochs

    def forward_step(self, data_obj: Data, mode: str):
        self.model.train() if mode == 'train' else self.model.eval()
        label, outs = data_obj.label, self.model(data_obj)
        loss, acc = self.loss_fun(outs, label), self.acc_fun(outs, label)
        
        self.log(f'{mode}_loss', loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=label.size(0))
        self.log(f'{mode}_acc', acc, on_step=True, on_epoch=True, sync_dist=True, batch_size=label.size(0))
        return loss

    def training_step(self, data_obj: Data, batch_idx: int):
        return self.forward_step(data_obj, 'train')
    
    def test_step(self, batch: Data, batch_idx: int):
        with torch.no_grad():
            return self.forward_step(batch, 'test')

    def validation_step(self, batch: Data, batch_idx: int):
        with torch.no_grad():
            return self.forward_step(batch, 'val')

    def compute_metric(self, trainer: pl.Trainer, test_loader: DataLoader, track: str):
        return trainer.test(self, dataloaders=test_loader, verbose=False)[0][f'test_{track}_epoch']
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.config.lr,
                                weight_decay=self.config.wd, eps=1e-07, amsgrad=True)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.config.gamma,
                                                             cooldown=self.config.cooldown)
        return [optimizer], {"scheduler": lr_scheduler, "interval": "epoch", "monitor": "train_loss"}
