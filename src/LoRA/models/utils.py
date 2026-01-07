import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

# set precision to what lightning suggests for this gpu
torch.set_float32_matmul_precision('high')


def save_checkpoint(name, model, optimizer, batch_size,
                    learning_rate, n_points, epoch):
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
    
    
def save_checkpoint_without_classifier_layer(name, model, optimizer, batch_size, learning_rate, n_points, epoch):
    
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

    
def plot_pc(sample_pc, i=0):
    # Convert to numpy array
    sample_pc_np = sample_pc.numpy()

    # Plot the 3D point cloud
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(sample_pc_np[i, :, 0], sample_pc_np[i, :, 1], sample_pc_np[i, :, 2], c=sample_pc_np[i, :, 2], s=30, marker='o', cmap="viridis", alpha=0.7)
    ax.set_zlim3d(-1, 1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

    
class DotDict(dict):
    """Dictionary with dot notation access"""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{name}'")

