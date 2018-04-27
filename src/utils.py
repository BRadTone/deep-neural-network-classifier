import numpy as np


def update_params(params, grads, learning_rate):

    L = len(params) // 2

    for l in range(L):
        params["W" + str(l + 1)] -= learning_rate * grads["dW" + str(l + 1)]
        params["b" + str(l + 1)] -= learning_rate * grads["db" + str(l + 1)]

    return params


def cost_fn(AL, Y):
    """
    Arguments:
    AL -- probability vector corresponding to your label predictions
    Y -- true "label" vector

    Returns:
    cost -- cross-entropy cost
    """

    m = Y.shape[1]
    cost = -1/m * np.sum(Y * np.log(AL) + (1-Y) * np.log(1- AL))
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    return cost
