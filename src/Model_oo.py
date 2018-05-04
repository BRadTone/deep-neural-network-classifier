import os
import sys
import pickle
import numpy as np
from src.Initializers import Initializers
from src.helpers.dnn_utils import ActivationFns, ActivationFnsDerivatives


class Model:
    ActivationFns = ActivationFns
    ActivationFnsDerivatives = ActivationFnsDerivatives

    def __init__(self, layers_dims=None, initializer=Initializers.he):
        self.params = []
        self.layers_dims = layers_dims

        if layers_dims:
            self.init_params(initializer)

    def predict(self):
        pass

    def train(self):
        pass

    def forward_prop(self, X):
        caches = []
        A = X
        L = len(self.layers_dims)

        for l in range(1, L):
            A_prev = A
            W = self.params['W' + str(l)]
            b = self.params['b' + str(l)]
            A, cache = self.forward(A_prev, W, b, ActivationFns.relu)
            caches.append(cache)

        AL, cache = self.forward(A, self.params['W' + str(L)], self.params['b' + str(L)], ActivationFns.sigmoid)
        caches.append(cache)

        assert AL.shape == (self.layers_dims[-1], X.shape[1])

        return AL, caches

    def forward(self, A_prev, W, b, activation):
        """
        Arguments:
        A_prev -- activations from previous layer (or input data)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- activation function callback, MUST return same size as its input

        Returns:
        A -- the output of the activation function, also called the post-activation value
        cache -- a python dictionary containing "linear_cache" and "activation_cache"
        """

        assert (hasattr(activation, '__call__'))

        Z = np.dot(W, A_prev) + b
        linear_cache = (A_prev, W, b)
        A = activation(Z)

        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, Z)

        return A, cache

    def cost(self):
        pass

    def back_prop(self):
        pass

    def load_params(self, path):
        try:
            with open(path, "rb") as f:
                self.params = pickle.load(f)
                self.update_layers_dims()
        except OSError as err:
            print(err)
            sys.exit()

    def save_params(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.params, f)

    def init_params(self, initializer):
        if initializer:
            self.params = initializer(self.layers_dims)
            return

        self.params = initializer(self.layers_dims)

    def update_layers_dims(self):
        # init new layer dims with first layer
        self.layers_dims = [self.params['W' + str(1)].shape[1]]

        for i in range(1, len(self.params) // 2 + 1):
            layer_units = self.params['W' + str(i)].shape[0]
            self.layers_dims.append(layer_units)
        assert len(self.layers_dims) - 1 == len(self.params) // 2




