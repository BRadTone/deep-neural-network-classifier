import glob
import os
import gzip
import pickle
import numpy as np
from trainings.utils import pickle_model, normalize, ints_to_binary_vec
from src.Model import Model

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
    Y = ints_to_binary_vec(train_y.reshape(1, m), 10)
    Y_valid = ints_to_binary_vec(valid_y.reshape(1, m_test), 10)

    X = normalize(train_x)
    X_valid = normalize(test_x)

    hyp_params = {
        'epochs': 2000,
        'learning_rate': 0.05,
        'layers_dims': [28 * 28, 100, 100, 10],
        'print_cost': True
    }

    model = Model(layers_dims=hyp_params['layers_dims'])
    model.train(X.T, Y, hyp_params['epochs'], hyp_params['learning_rate'], hyp_params['print_cost'], X_valid.T, Y_valid)

    pickle_name = '../datasets/mnist/model-alpha-{0}-iterations-{1}-layers-{2}.pickle' \
        .format(hyp_params['learning_rate'], hyp_params['epochs'], str(hyp_params['layers_dims']))

    pickle_model(pickle_name, model)
