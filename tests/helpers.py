import pickle
import numpy as np


def pickle_model(path, model):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def normalize(x):
    return (x - np.mean(x)) / np.std(x)
