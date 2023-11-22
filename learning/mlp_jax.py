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
    num_hidden: list
    num_outputs: int

    def setup(self):
        self.linear = [nn.Dense(features=self.num_hidden[i]) for i in range(len(self.num_hidden))]
        self.linear2 = nn.Dense(features=self.num_outputs)

    def __call__(self, x):
        for i in range(len(self.num_hidden)):
            x = self.linear[i](x)
            x = nn.elu(x)
        x = self.linear2(x)
        return x
