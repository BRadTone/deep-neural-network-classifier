import pickle
import numpy as np
from src.forward_prop import model_forward


def pickle_model(path, model):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def normalize(x):
    return (x - np.mean(x)) / np.std(x)


def predict(x, params):
    AL, _ = model_forward(x, params)

    return AL