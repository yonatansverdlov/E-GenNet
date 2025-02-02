"""
Utils file.
"""
import datetime
import os
import shutil
from pathlib import Path
from typing import Dict, Tuple, List

import lightning as pl
import torch
import yaml
from easydict import EasyDict
from torch_geometric.loader import DataLoader


from data_creation.Datasets import BatchDataSet, k_chain_dataset, hard_dataset
from data_creation.chemical_datasets.BDE import BDE
from data_creation.chemical_datasets.Drugs import Drugs
from data_creation.chemical_datasets.Kraken import Kraken
from model.lightning_model import ModelWrapped
from model.models import GenericNet
from lightning.pytorch.callbacks import ModelCheckpoint


class BasicTrainCallback(pl.Callback):
    """
    Callback created for shuffling every epoch.
    """

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        On the training epoch end we shuffle the dataset.
        Args:
            trainer: The trainer.
            pl_module: The module.

        """
        epochs = trainer.current_epoch
        if epochs % pl_module.reshuffle_every_n_epochs == 0:
            trainer.train_dataloader.reshuffle_grouped_dataset()


def return_dataloader(config: EasyDict, types: str, path_to_project: Path) -> Dict:
    """
    Given config, and data type returns the dataloader.
    Args:
        config: The config.
        types: The task type.
        path_to_project: The path to project.

    Returns: The dataloader.

    """
    if types == 'k_chain':
        dataloaders = return_classification_loader(config=config)
    elif types == 'Hard': 
        dataloaders = return_hard_loader(config = config) 
    else:
        dataloaders = return_drugs_loaders(config=config,
                                           path_to_project=path_to_project)

    config.type_config.task_specific.mean = dataloaders.mean
    config.type_config.task_specific.std = dataloaders.std

    return dataloaders


def return_model_path(config: EasyDict, task: str) -> Tuple[Path, str]:
    """
    Return the model path.
    Args:
        config: The config dictionary.
        task: The task.

    Returns:

    """
    # The path to project.
    path_to_project = Path(os.path.abspath(__file__)).parent
    # Init.
    model_path = 'Model_best'
    # Add all params.
    # Path to the model dir.
    model_dir = os.path.join(path_to_project,
                             f'data/models/{config.type}/{task}/'
                             f'{model_path}')
    print(f"Saving into: {model_dir}")
    # Save the code our_exps.
    return path_to_project, model_dir


def return_callbacks(model_dir: str, metric_track: str,
                     types: str, save_top_k: int = 10) -> List:
    """
    Return callbacks.
    Args:
        model_dir: The model dir.
        metric_track: The metric track.
        types: The data type.
        save_top_k: The save top k.

    Returns: The Callbacks.

    """
    # The callbacks.
    # The model checkpoint.
    mode = 'max' if metric_track == 'acc' and types in ['k_chain','Hard'] else 'min'
    callbacks = [ModelCheckpoint(dirpath=model_dir,
                                              filename='{epoch}-f{val_loss:.5f}' if metric_track == 'loss' else
                                              '{epoch}-f{val_acc:.5f}',
                                              save_top_k=save_top_k,
                                              monitor=f'val_{metric_track}',
                                              save_last=True, mode=mode)]

    # For drugs shuffling.
    if types not in ['k_chain','Hard']:
        callbacks.append(BasicTrainCallback())
    return callbacks


def return_trainer_data_and_model(config: EasyDict, types: str, metric_track: str, task: str) -> Tuple:
    """
    Args:
        config: The config file
        types: The point type.
        metric_track: The metric to track.
        task: The task.

    Returns: Trains and return the Accuracy of the trained model.

    """
    # path to the project.
    path_to_project, model_dir = return_model_path(config=config, task=task)
    # The data-loader.
    dataloaders = return_dataloader(config=config, types=types,
                                    path_to_project=path_to_project)
    # Our model.
    model = GenericNet(config=config)
    # The wrapper.
    wrapped_model = ModelWrapped(model=model, config=config)
    # Checkpoint plugin.
    if config.general_config.enable_checkpointing:
        callbacks = return_callbacks(model_dir=model_dir, metric_track=metric_track,
                                     types=types, save_top_k=config.general_config.save_top_k)
    else:
        callbacks = [[]]
    # The trainer.
    trainer = pl.Trainer(default_root_dir=os.path.join(path_to_project, 'data'),
                         accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                         max_epochs=config.general_config.max_epochs,
                         callbacks=callbacks,
                         check_val_every_n_epoch=config.type_config.common_to_all_tasks.check_val_every_n_epoch,
                         enable_model_summary=config.general_config.enable_model_summary,
                         enable_progress_bar=config.general_config.enable_progress_bar,
                         precision=config.type_config.common_to_all_tasks.precision,
                         accumulate_grad_batches=config.type_config.task_specific[task].accumulate_grad_batches,
                         enable_checkpointing=config.general_config.enable_checkpointing,
                         )
    return wrapped_model, trainer, dataloaders, callbacks[0]


def train_model(config: EasyDict, types: str, metric_track: str, task: str,
                train=True):
    """

    Args:
        config: The config dictionary.
        types: The types.
        metric_track: The metric to track.
        task: The task.
        train: Whether to train.

    Returns: The model, trainer, checkpoint.

    """
    wrapped_model, trainer, dataloaders, ckpt = return_trainer_data_and_model(config=config, types=types,
                                                                              metric_track=metric_track, task=task)

    train_dataloader = dataloaders.train_dl
    val_dataloader = dataloaders.val_dl
    current_time = datetime.datetime.now()
    if train:
        trainer.fit(wrapped_model, train_dataloader, val_dataloader)
    time_after_training = datetime.datetime.now()
    print(f"Time for training: {(time_after_training - current_time).total_seconds() / 60.0}")
    # Return accuracy.
    return wrapped_model, trainer, dataloaders, ckpt


def return_classification_loader(config: EasyDict) -> EasyDict:
    """
    Given tuple type, we return the dataloader.
    Args:
        config: The config file.


    Returns: The dataloader.

    """
    dataloader = DataLoader(k_chain_dataset(config=config))

    dataloaders = {'train_dl': dataloader, 'test_dl': dataloader, 'val_dl': dataloader, 'mean': .0,
                   'std': 1.0}

    return EasyDict(dataloaders)

def return_hard_loader(config: EasyDict) -> EasyDict:
    """
    Given tuple type, we return the dataloader.
    Args:
        config: The config file.


    Returns: The dataloader.

    """
    dataloader = DataLoader(hard_dataset(config=config))

    dataloaders = {'train_dl': dataloader, 'test_dl': dataloader, 'val_dl': dataloader, 'mean': .0,
                   'std': 1.0}

    return EasyDict(dataloaders)

def return_drugs_loaders(config: EasyDict, path_to_project: Path) -> EasyDict:
    """
    Returns drug loader.
    Args:
        config: The config dictionary.
        path_to_project: The project path.

    Returns:

    """
    # The task.
    task = config.task
    # The generator.
    generator = torch.Generator().manual_seed(config.type_config.common_to_all_tasks.seed)
    # Data mappings.
    dataset_dict = {'Drugs': Drugs, 'Kraken': Kraken, 'BDE': BDE}
    # The all samples dataset.
    full_dataset = dataset_dict[config.type](root=os.path.join(path_to_project, 'dataset/'))
    # Size splits.
    train_size, val_size = (config.type_config.common_to_all_tasks.train_size,
                            config.type_config.common_to_all_tasks.val_size)
    # Test size.
    test_size = 1 - train_size - val_size
    # Split.
    train_set, val_set, test_set = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size],
                                                                 generator=generator)
    mean, std = full_dataset.get_mean(target=task), full_dataset.get_std(target=task)
    # Make train DL.
    train_dl = BatchDataSet(train_set, batch_size=config.type_config.task_specific[task].bs, task=task,
                            descriptors=full_dataset.descriptors)
    # Make val DL.
    val_dl = BatchDataSet(val_set, batch_size=2 * config.type_config.task_specific[task].bs,
                          task=task,
                          descriptors=full_dataset.descriptors)
    # Make test DL.
    test_dl = BatchDataSet(test_set, batch_size=2 * config.type_config.task_specific[task].bs,
                           task=task,
                           descriptors=full_dataset.descriptors)
    print("Done splitting to train, test, val.")
    output = EasyDict({'train_dl': train_dl, 'test_dl': test_dl, 'val_dl': val_dl, 'mean': mean, 'std': std})
    # Return the data dictionary.
    return output


def number_of_params(model: torch.nn.Module) -> int:
    """
    Returns the number of parameters of module.
    Args:
        model: The model.

    Returns: The number of parameters.

    """
    # Sum over all parameters according to their dimension.
    return sum(p.numel() for p in model.parameters())


def train_type_n_times(task: str, types: str, fix_seed: bool = False,batch_size = 512,accum_grad = 1,
                       metric_track='acc', train: bool = True, num_times: int = 1,epochs = 1500) -> torch.float:

    """

    Args:
        task: The task.
        types: The tuple type.
        metric_track: What to track, acc/loss.
        train: Whether train/test.
    Returns: The accuracy overall seeds.

    """
    test_acc, val_acc, train_acc = .0, .0, .0
    path = Path(os.path.abspath(__file__)).parent
    with open(os.path.join(path, f'data_creation/config_files/{types}_config.yaml')) as f:
        type_config = EasyDict(yaml.safe_load(f)[types])
    with open(os.path.join(path, f'data_creation/config_files/General_config.yaml')) as f:
        general_config = EasyDict(yaml.safe_load(f)['General_config'])
    print("Loaded the config files!")

    for i in range(num_times):
        config = EasyDict({'type_config': type_config, 'general_config': general_config, 'type': types, 'task': task})
        if fix_seed:                
            config.type_config.common_to_all_tasks.seed = i
        config.general_config.max_epochs = epochs
        config.type_config.task_specific[task].bs = batch_size
        config.type_config.task_specific[task].accumulate_grad_batches = accum_grad
        
        wrapped_model, trainer, dataloaders, ckpt = train_model(types=types, metric_track=metric_track,
                                                                config=config,
                                                                train=train, task=task)
        # Load the best model and test it.
        checkpoint = torch.load(ckpt.best_model_path)
        wrapped_model.load_state_dict(checkpoint['state_dict'])

        test_acc += wrapped_model.compute_metric(trainer=trainer, test_loader=dataloaders.test_dl, track='acc')
        val_acc += wrapped_model.compute_metric(trainer=trainer, test_loader=dataloaders.val_dl, track='acc')
        train_acc += wrapped_model.compute_metric(trainer=trainer, test_loader=dataloaders.train_dl, track='acc')

    return test_acc / num_times, val_acc / num_times, train_acc / num_times

def return_sets(task: str, types: str, fix_seed: bool = False,batch_size = 512,accum_grad = 1,
                       metric_track='acc', train: bool = True, num_times: int = 1,epochs = 1500) -> torch.float:

    """

    Args:
        task: The task.
        types: The tuple type.
        metric_track: What to track, acc/loss.
        train: Whether train/test.
    Returns: The accuracy overall seeds.

    """
    path = Path(os.path.abspath(__file__)).parent
    with open(os.path.join(path, f'data/config_files/{types}_config.yaml')) as f:
        type_config = EasyDict(yaml.safe_load(f)[types])
    with open(os.path.join(path, f'data/config_files/General_config.yaml')) as f:
        general_config = EasyDict(yaml.safe_load(f)['General_config'])
    print("Loaded the config files!")

    for i in range(num_times):
        config = EasyDict({'type_config': type_config, 'general_config': general_config, 'type': types, 'task': task})
        if fix_seed:
            config.type_config.common_to_all_tasks.seed = i
        config.general_config.max_epochs = epochs
        config.type_config.task_specific[task].bs = batch_size
        config.type_config.task_specific[task].accumulate_grad_batches = accum_grad
        
        wrapped_model, trainer, dataloaders, ckpt = train_model(types=types, metric_track=metric_track,
                                                                config=config,
                                                                train=False, task=task)
        return dataloaders

