import numpy as np
from src.helpers.dnn_utils import ActivationFns


def model_forward(X, parameters):
    """
   Returns:
   AL -- the output of the activation function form last layer.
   caches -- a python dictionary containing "linear_cache" and "activation_cache" for each layer
   """

    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]

        A, cache = forward(A_prev, W, b, ActivationFns.relu)
        caches.append(cache)

    AL, cache = forward(A, parameters['W' + str(L)], parameters['b' + str(L)], ActivationFns.sigmoid)
    caches.append(cache)

    assert (AL.shape == (1, X.shape[1]))

    return AL, caches


def forward(A_prev, W, b, activation):
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
