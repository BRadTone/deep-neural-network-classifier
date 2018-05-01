import glob
import os
import gzip
import pickle
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from tests.utils import pickle_model, normalize
from src.model import model

# load data
with gzip.open('../datasets/mnist/mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set


if __name__ == '__main__':
    m = train_x.shape[0]
    m_test = test_x.shape[0]

    # map train_y to output vector
    Y = np.zeros((10, m))
    Y_valid = np.zeros((10, m_test))

    Y[train_y.reshape(1, m), np.arange(m)] = 1
    Y_valid[test_y.reshape(1, m_test), np.arange(m_test)] = 1

    X = normalize(train_x)
    X_valid = normalize(test_x)

    hyp_params = {
        'epochs': 2000,
        'learning_rate': 0.1,
        'layers_dims': [28 * 28, 50, 50, 10],
        'print_cost': True
    }

    model = model(X.T, Y, X_valid.T, Y_valid, **hyp_params)

    pickle_name = '../datasets/mnist/model-alpha-{0}-iterations-{1}-layers-{2}.pickle' \
        .format(hyp_params['learning_rate'], hyp_params['epochs'], str(hyp_params['layers_dims']))

    pickle_model(pickle_name, model)
