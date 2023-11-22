"""
SYNOPSIS
    Contains functions required to load data, train and run inference on
    neural network models for learning value function from trajectories.

DESCRIPTION
    Implementation uses JAX libraries (flax) to define functions to train,
    evaluate and save deep learning models.

AUTHOR
    Anusha Srikanthan <sanusha@seas.upenn.edu>

VERSION
    0.0
"""
from flax import linen as nn
from flax.training import checkpoints  # need to install tensorflow
import torch.utils.data as data
import numpy as np
import optax

## Progress bar
from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
from torch.utils.data import Dataset
import torch

from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter()


class TrajDataset(Dataset):
    """
    Dataset class inherited from torch modules
    """
    # def __init__(self, file_path, device=torch.device('cpu'), transform=None, target_transform=None):
    def __init__(self, file_path, transform=None, target_transform=None):
        """
        Creating the dataset class for our pipeline
        :param file_path: path of csv file
        :param costs: an array containing the costs for each trajectory
        :param transform: function to transform the coefficient data, if needed
        :param target_transform: function to transform the costs such as normalization, if needed
        """
        self.data = np.loadtxt(file_path,delimiter=",",skiprows=1,)

        # Assuming the first column is 'traj_number', the second is 'cost', and the rest are coefficients
        self.coeffs = self.data[:, 2:]  # All coefficient columns
        self.costs = self.data[:, 1]  # Cost column
        # self.coeffs = torch.tensor(self.data[:, 2:], dtype=torch.float32, device=device)
        # self.costs = torch.tensor(self.data[:, 1], dtype=torch.float32, device=device)

        self.transform = transform
        self.target_transform = target_transform
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # The first column is 'traj_number' and the second column is 'cost'
        # The rest of the columns are coefficients: x_poly_seg_0_coeff_0,x_poly_seg_0_coeff_1, ..., yaw_poly_seg_2_coeff_7
        coeffs = self.coeffs[idx]
        cost = self.costs[idx]
        # if self.transform:
        #     coeffs = self.transform(coeffs)
        # if self.target_transform:
            # cost = self.target_transform(cost)

        return coeffs, cost

    def num_coefficients(self):
        return self.coeffs.shape[1] if len(self.coeffs) > 0 else 0



def numpy_collate(batch):
    """
    A numpy helper function for efficient batching from JAX documentation
    :param batch: batches from the dataset
    :return: batch samples
    """
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def calculate_loss(state, params, batch):
    """
    Loss function for training defined as the l2 loss between prediction and target
    :param state:
    :param params:
    :param batch:
    :return:
    """
    data_coeffs, data_cost = batch
    pred = state.apply_fn(params, data_coeffs)
    target = data_cost

    # Calculate the loss
    loss = optax.l2_loss(pred.ravel(), target.ravel()).mean()

    return loss


@jax.jit  # Jit the function for efficiency
def train_step(state, batch):
    """
    One iteration of training by running the back propagation through the batch
    :param state: weights of the neural network model
    :param batch: batch from the dataset
    :return: state, loss
    """
    # Gradient function
    grad_fn = jax.value_and_grad(
        calculate_loss,  # Function to calculate the loss
        argnums=1,  # Parameters are second argument of the function
        has_aux=False,  # Function has additional outputs, here accuracy
    )
    # Determine gradients for current model, parameters and batch
    loss, grads = grad_fn(state, state.params, batch)
    print(loss)
    # Perform parameter update with gradients and optimizer
    state = state.apply_gradients(grads=grads)
    # Return state and any other value we might want
    return state, loss


@jax.jit  # Jit the function for efficiency
def eval_step(state, batch):
    """
    One iteration of predictions to evaluate the model
    :param state: weights of the neural network model
    :param batch: batch from the dataset
    :return: loss
    """
    # Determine the accuracy
    loss = calculate_loss(state, state.params, batch)
    return loss


def train_model(state, data_loader, num_epochs=100):
    """
    Train the model over the training dataset
    :param state: weights of the neural network model
    :param data_loader: batched dataset
    :param num_epochs: number of epochs
    :return: state
    """
    # Training loop
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0
        for batch in data_loader:
            state, loss = train_step(state, batch)
            # Accumulate the loss over the epoch
            epoch_loss += loss

        # Record the epoch loss at the end of the epoch
        writer.add_scalar('Train loss', np.array(epoch_losss), epoch)
        
    return state


def eval_model(state, data_loader, batch_size):
    """
    Evaluate model over the test dataset
    :param state: weights of the neural network model
    :param data_loader: batched dataset
    :param batch_size: number of samples in a batch
    :return: None
    """
    all_losses, batch_sizes = [], []
    for batch in data_loader:
        batch_loss = eval_step(state, batch)
        all_losses.append(batch_loss)
        batch_sizes.append(batch[0].shape[0])
    # Weighted average since some batches might be smaller
    loss = sum([a * b for a, b in zip(all_losses, batch_sizes)]) / sum(batch_sizes)
    # writer.add_scalar("Train batch loss", np.array(loss), count)
    print(f"Loss of the model: {loss:4.2f}")


def restore_checkpoint(state, workdir):
    """
    Restore the weights of the model
    :param state: initialized model object
    :param workdir: file path
    :return: state of the network
    """
    return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir, step=0):
    """
    Save the weights to a file
    :param state: model object
    :param workdir: file path
    :param step: checkpoint index
    :return: None
    """
    checkpoints.save_checkpoint(workdir, state, step)
    # checkpoints.save_checkpoint(workdir, state, step, overwrite=True, keep=2)






