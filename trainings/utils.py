import pickle
import numpy as np
from src.ForwardProp import ForwardProp


def pickle_model(path, model):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def normalize(x):
    return (x - np.mean(x)) / np.std(x)


def predict(x, params):
    AL, _ = ForwardProp.model_forward(x, params)
    AL_exps = np.exp(AL)
    return AL_exps / np.sum(AL_exps, axis=0)


def accuracy(x, y, params):
    """

    :param x: (f,m) ndarray
    :param y: (1,m) ndarray
    :param params: dict of matrices (W, b)
    """
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert x.shape[1] == y.shape[1]

    m = x.shape[1]

    AL = predict(x, params)
    AL_1D = np.argmax(AL, axis=0).reshape(1, m)

    return np.sum(AL_1D == y) / m


def ints_to_binary_vec(y, classes):
    """

    :param y:  (1,m) ndarray
    :param classes: int: clf outputs
    :return: binary matrix
    """
    assert isinstance(y, np.ndarray)
    m = y.shape[1]
    Y = np.zeros(shape=(classes, m))
    Y[y.reshape(1, m), np.arange(m)] = 1

    return Y
