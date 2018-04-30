import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tests.mnist import test_x, test_y, train_y, train_x
from tests.utils import normalize, predict, accuracy, ints_to_binary_vec
from src.utils import cost_fn

base_path = '../datasets/mnist'
path = base_path + '/model-alpha-0.1-iterations-2000-layers-3.pickle'
params = {}
if os.path.isfile(path):
    params, costs = pickle.load(open(path, 'rb'))

X_test = normalize(test_x).T  # (n, m)
Y_test = test_y.reshape(1, -1)  # (1, m)

acc = accuracy(X_test, Y_test, params)
print(acc)

AL_test = predict(X_test, params)
cost_test = cost_fn(AL_test, ints_to_binary_vec(Y_test, 10))

print(cost_test)
# idx = 444
# aL = predict(X[idx].reshape(784, 1), params)
# plt.imshow(test_x[idx].reshape((28, 28)))
# plt.show()
