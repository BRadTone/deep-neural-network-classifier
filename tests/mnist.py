import glob
import os
import gzip
import pickle
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from tests.helpers import pickle_model, normalize
from src.model import model


# load data
with gzip.open('../datasets/mnist/mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set


# show image
# img_idx = 23
# plt.imshow(train_x[img_idx].reshape((28, 28)), cmap=cm.Greys_r)
# plt.show()


if __name__ == "__main__":
    m = train_x.shape[0]

    # map train_y to output vector
    Y = np.zeros((10, m))
    Y[train_y.reshape(1, m), np.arange(m)] = 1
    X = normalize(train_x)

    hyp_params = {
        'epochs': 150,
        'learning_rate': 0.18,
        'layers_dims': [28 * 28, 100, 100, 10],
        'print_cost': True
    }

    model = model(X.T, Y, **hyp_params)

    pickle_name = '../datasets/mnist/model-alpha-{0}-iterations-{1}.pickle'\
        .format(hyp_params['learning_rate'], hyp_params['epochs'])

    pickle_model(pickle_name, model)


