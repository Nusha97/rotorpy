"""
SYNOPSIS
    Implementation of input convex neural network using JAX libraries
DESCRIPTION

    Preliminary implementation of ICNN using JAX libraries
AUTHOR

    Anusha Srikanthan <sanusha@seas.upenn.edu>
LICENSE

VERSION
    0.0
"""

from flax import linen as nn
import torch.utils.data as data
import numpy as np
import jax.numpy as jnp
import jax
from functools import partial


def Initialize_W_B(key, layer_units_c, input_features_c, seed, last_layer=False):
    """initialize weights and biases
    Input:
    key:
    layer_units:list of dense layer units"""


    k = input_features_c
    minval = 0.0
    maxval = 1.0

    if last_layer:
        wz = jax.random.uniform(key=seed, shape=(input_features_c, layer_units_c), minval=minval/k, maxval=maxval/k,
                                dtype=jnp.float32)
        # bias vector
        b = jax.random.uniform(key=seed, minval=minval/k, maxval=maxval/k, shape=(layer_units_c,), dtype=jnp.float32)
        return [wz, b]

    else:
        weights_and_bias = []
        for i, units in enumerate(layer_units_c):

            # first layer weight has dim (num_units, input shape)
            if i == 0:
                wz = jax.random.uniform(key=seed, shape=(input_features_c, units), minval=minval/k, maxval=maxval/k,
                                        dtype=jnp.float32)
                wy = jax.random.uniform(key=seed, shape=(input_features_c, units), minval=minval/k, maxval=maxval/k,
                                        dtype=jnp.float32)

            # if not first layer
            else:
                layer_sq = layer_units_c[i-1]
                wz = jax.random.uniform(key=seed, shape=(layer_units_c[i - 1], units), minval=minval/layer_sq, maxval=maxval/layer_sq,
                                        dtype=jnp.float32)
                wy = jax.random.uniform(key=seed, shape=(input_features_c, units), minval=minval/k, maxval=maxval/k,
                                        dtype=jnp.float32)

            # bias vector
            b = jax.random.uniform(key=seed, minval=minval/(units * 5), maxval=maxval/(units * 5), shape=(units,), dtype=jnp.float32)

            # append weights
            weights_and_bias.append([wz, wy, b])

        return weights_and_bias


def mysoftplus(x):
    ones = jnp.ones(x.shape)
    y = jnp.log(ones + jnp.exp(x))
    return y


class FICNN(nn.Module):
    num_hidden_c: list
    num_outputs: int
    input_features_c: int
    seed: int

    def setup(self):
        self.linear_c = self.param('layer_c', Initialize_W_B, self.num_hidden_c, self.input_features_c, self.seed)
        self.linear2 = self.param('layer_2', Initialize_W_B, self.num_outputs, self.num_hidden_c[-1], self.seed, True)


    def __call__(self, y):
        """
        Call function to propagate the input through the weights of the neural network
        :param x:
        :param y:
        :return:
        """
        #import pdb;
        #pdb.set_trace()
        # breakpoint()
        z = nn.relu(self.linear_c[0][0]).T @ y.T + self.linear_c[0][2][:, None]
        # z = nn.relu(self.linear_c[0][0]).T @ y + self.linear_c[0][2]
        z = nn.relu(z)
        for i in range(0, len(self.num_hidden_c)-1):
            # print("loop iter", i)
            z = nn.relu(self.linear_c[i+1][0]).T @ z + self.linear_c[i+1][1].T @ y.T + self.linear_c[i+1][2][:, None]
            z = nn.relu(z)
        z = nn.relu(self.linear2[0]).T @ z + self.linear2[1]
        # z = nn.relu(z)
        return z