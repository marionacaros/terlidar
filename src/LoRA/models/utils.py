import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

# set precision to what lightning suggests for this gpu
torch.set_float32_matmul_precision('high')


def save_checkpoint(name: str, model: nn.Module, optimizer: torch.optim.Optimizer, batch_size: int,
                    learning_rate: float, n_points: int, epoch: int) -> None:
    """
    Save a training checkpoint to ``<name>.pt``.

    The file is a dict with keys ``model`` and ``optimizer`` (state dicts), ``batch_size``,
    ``lr``, ``number_of_points`` and ``epoch``.

    :param name: output path without the ``.pt`` extension
    """
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'batch_size': batch_size,
        'lr': learning_rate,
        'number_of_points': n_points,
        'epoch':epoch
    }
    filename = name + '.pt'
    torch.save(state, filename)
    
    
def save_checkpoint_without_classifier_layer(name: str, model: nn.Module, optimizer: torch.optim.Optimizer,
                                             batch_size: int, learning_rate: float, n_points: int,
                                             epoch: int) -> None:
    """
    Same as ``save_checkpoint`` but without ``classifier.weight`` and ``classifier.bias`` in the
    model state dict, so the checkpoint can initialise a model with a different number of classes.
    The model must have a layer named ``classifier`` (``PointNet2``).

    :param name: output path without the ``.pt`` extension
    """
    
    state_dict=model.state_dict()
    # Remove the last layer from the state dictionary
    del state_dict['classifier.weight']
    del state_dict['classifier.bias']
    
    state = {
        'model': state_dict,
        'optimizer': optimizer.state_dict(),
        'batch_size': batch_size,
        'lr': learning_rate,
        'number_of_points': n_points,
        'epoch':epoch
    }
    filename = name + '.pt'
    torch.save(state, filename)

    

    
