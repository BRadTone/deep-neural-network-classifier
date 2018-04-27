import glob
import os
import gzip
import pickle
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from src.model import model

with gzip.open('../datasets/mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

train_x, train_y = train_set
m = train_x.shape[0]
# plt.imshow(train_x[9].reshape((28, 28)), cmap=cm.Greys_r)
# print(train_y)
# plt.show()

# print(train_x.shape)
# print(type(train_x))

model_shape = [28 * 28, 10, 10, 10]

y = np.zeros((10, m))
y[train_y.reshape(1, m) - 1, np.arange(m)] = 1

model(train_x.T, y, layers_dims=model_shape, print_cost=True)



# todo https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
# pickle_name = '../datasets/mnist_model_params.pickle'
#
# # if os.path.isfile(pickle_name) and not debug_csv:
# #     return pd.read_pickle(pickle_name)
#
# print('creating pickle...')
#
# model_params = None
#
# pickle_out = open(pickle_name, 'wb')
# pickle.dump(model_params, pickle_out)
# pickle_out.close()
