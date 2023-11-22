"""
SYNOPSIS
    Implementation of input convex neural network using JAX libraries
DESCRIPTION

    Preliminary implementation of PICNN using JAX libraries
AUTHOR

    Anusha Srikanthan <sanusha@seas.upenn.edu>
LICENSE

VERSION
    0.0
"""

from flax import linen as nn
import jax.numpy as jnp
import jax


def Initialize_W_B(key, layer_units_c, layer_units_nc, input_features_nc, input_features_c, seed, last_layer=False):
    """initialize weights and biases
    Input:
    key:
    layer_units:list of dense layer units"""


    k = input_features_c
    minval = -1.0
    maxval = 1.0

    if last_layer:
        layer_sq = layer_units_c  # num_last_convex_output for z
        layer_nsq = input_features_nc  # num_last_nonvex_output for u
        units = layer_units_nc  # num_outputs

        zuu = jax.random.uniform(key=seed, shape=(layer_sq, layer_nsq), minval=minval / layer_sq,
                                 maxval=maxval / layer_sq,
                                 dtype=jnp.float32)
        zzu = jax.random.uniform(key=seed, shape=(units, layer_sq), minval=minval / units, maxval=maxval / units,
                                 dtype=jnp.float32)
        yuu = jax.random.uniform(key=seed, shape=(input_features_c, layer_nsq), minval=minval / input_features_c,
                                 maxval=maxval / input_features_c,
                                 dtype=jnp.float32)
        zyu = jax.random.uniform(key=seed, shape=(units, input_features_c), minval=minval / units,
                                 maxval=maxval / units,
                                 dtype=jnp.float32)
        zu = jax.random.uniform(key=seed, shape=(units, layer_nsq), minval=minval / units, maxval=maxval / units,
                                dtype=jnp.float32)
        # bias vector
        b = jax.random.uniform(key=seed, minval=minval/k, maxval=maxval/k, shape=(units,), dtype=jnp.float32)
        # return [wz, b]
        return [zuu, zzu, yuu, zyu, zu, b]

    else:
        weights_and_bias = []
        n = input_features_nc
        for i, units in enumerate(layer_units_c):

            # first layer weight has dim (num_units, input shape)
            if i == 0:
                zuu = 0.0
                zzu = 0.0
                yuu = jax.random.uniform(key=seed, shape=(input_features_c, input_features_nc), minval=minval / input_features_c, maxval=maxval/input_features_c,
                                         dtype=jnp.float32)
                zyu = jax.random.uniform(key=seed, shape=(units, input_features_c), minval=minval / units, maxval = maxval / units,
                                         dtype=jnp.float32)
                zu = jax.random.uniform(key=seed, shape=(units, input_features_nc), minval=minval/units, maxval=maxval/units,
                                        dtype=jnp.float32)

            # if not first layer
            else:
                layer_sq = layer_units_c[i-1]
                layer_nsq = layer_units_nc[i - 1]

                zuu = jax.random.uniform(key=seed, shape=(layer_sq, layer_nsq), minval=minval / layer_sq, maxval=maxval / layer_sq,
                                         dtype=jnp.float32)
                zzu = jax.random.uniform(key=seed, shape=(units, layer_sq), minval=minval/units, maxval=maxval/units,
                                         dtype=jnp.float32)
                yuu = jax.random.uniform(key=seed, shape=(input_features_c, layer_nsq), minval=minval/input_features_c, maxval=maxval/input_features_c,
                                         dtype=jnp.float32)
                zyu = jax.random.uniform(key=seed, shape=(units, input_features_c), minval=minval / units, maxval = maxval / units,
                                         dtype=jnp.float32)
                zu = jax.random.uniform(key=seed, shape=(units, layer_nsq), minval=minval / units, maxval=maxval/units,
                                        dtype=jnp.float32)

            # bias vector
            b = jax.random.uniform(key=seed, minval=minval/(units * 5), maxval=maxval/(units * 5), shape=(units,), dtype=jnp.float32)


            # append weights
            weights_and_bias.append([zuu, zzu, yuu, zyu, zu, b])

        return weights_and_bias


class PICNN(nn.Module):
    num_hidden_nc: list
    num_hidden_c: list
    num_outputs: int
    input_features_nc: int
    input_features_c: int
    seed: int


    def setup(self):
        self.linear_nc = [nn.Dense(features=self.num_hidden_nc[i]) for i in range(len(self.num_hidden_nc))]
        self.linear_c = self.param('layer_c', Initialize_W_B, self.num_hidden_c, self.num_hidden_nc, self.input_features_nc, self.input_features_c, self.seed)
        self.linear_last = self.param('layer_last', Initialize_W_B, self.num_hidden_c[-1], self.num_outputs, self.num_hidden_nc[-1], self.input_features_c, self.seed, True)


    #@nn.compact
    #def __call__(self, x, y, train):
    def __call__(self, x, y):
        """
        Call function to propagate the input through the weights of the neural network
        :param x:
        :param y:
        :return:
        """

        yuu = self.linear_c[0][2] @ x.T
        zyu = self.linear_c[0][3] @ jnp.multiply(yuu.T, y).T
        zu = self.linear_c[0][4] @ x.T
        z = zyu + zu + self.linear_c[0][5][:, None]
        u = x
        for i in range(1, len(self.num_hidden_nc)):
            #u_next = nn.BatchNorm(use_running_average=not train, momentum=0.9, epsilon=1e-5, dtype=jnp.float32)(u)
            #u_next = self.linear_nc[i-1](u_next)
            u_next = self.linear_nc[i - 1](u)
            u_next = nn.relu(u_next)

            zuu = nn.relu(self.linear_c[i][0] @ u_next.T)
            zzu = self.linear_c[i][1] @ jnp.multiply(zuu, z)
            yuu = self.linear_c[i][2] @ u_next.T
            zyu = self.linear_c[i][3] @ jnp.multiply(yuu.T, y).T
            zu = self.linear_c[i][4] @ u_next.T
            z = zzu + zyu + zu + self.linear_c[i][5][:, None]
            u = u_next
        u_next = self.linear_nc[-1](u)
        u_next = nn.relu(u_next)
        #import pdb;
        #pdb.set_trace()
        zuu = nn.relu(self.linear_last[0] @ u_next.T)
        zzu = self.linear_last[1] @ jnp.multiply(zuu, z)
        yuu = (self.linear_last[2] @ u_next.T)[-1, None]
        zyu = self.linear_last[3] @ jnp.multiply(yuu.T, y).T
        zu = self.linear_last[4] @ u_next.T
        z = zzu + zyu + zu + self.linear_last[5][:, None]
        return z




