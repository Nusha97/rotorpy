"""
SYNOPSIS
    Implementation of multilayer perceptron network using JAX libraries
DESCRIPTION

    Contains one module:
    a) MLP - defines the layers and depth of the multilayer perceptron
AUTHOR

    Anusha Srikanthan <sanusha@seas.upenn.edu>
LICENSE

VERSION
    0.0
"""

from flax import linen as nn


class MLP(nn.Module):
    num_hidden: list  # List of numbers indicating the number of neurons in each hidden layer
    num_outputs: int  # Number of neurons in the output layer

    @nn.compact
    def __call__(self, x):
        # Define the hidden layers
        for hidden_units in self.num_hidden:
            x = nn.Dense(features=hidden_units)(x)  # Create a dense layer with the specified number of units
            x = nn.relu(x)  # Apply the ReLU activation function

        # Define the output layer
        x = nn.Dense(features=self.num_outputs)(x)  # Create the output dense layer
        return x