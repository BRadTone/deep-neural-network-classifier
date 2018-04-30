import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tests.mnist import test_x, test_y, train_y, train_x
from tests.utils import normalize, predict
from src.utils import cost_fn

base_path = '../datasets/mnist'
path = base_path + '/model-alpha-0.1-iterations-2000-layers-3.pickle'
params = {}
if os.path.isfile(path):
    params, costs = pickle.load(open(path, 'rb'))

m = train_x.shape[0]
m_test = test_x.shape[0]

Y = np.zeros((10, m))
Y_test = np.zeros((10, m_test))
Y[train_y.reshape(1, m), np.arange(m)] = 1
Y_test[test_y.reshape(1, m_test), np.arange(m_test)] = 1

X = normalize(train_x)
X_test = normalize(test_x)

AL = predict(X.T, params)
AL_test = predict(X_test.T, params)

AL_test_1D = np.argmax(AL_test, axis=0).reshape(1, m_test)
Y_test_1D = test_y.reshape(1, m_test)

diff = AL_test_1D == Y_test_1D

accuracy_test = np.sum(diff) / m_test
print(accuracy_test)

idx = 444
aL = predict(X[idx].reshape(784, 1), params)

cost_train = cost_fn(AL, Y)
cost_test = cost_fn(AL_test, Y_test)

# plt.imshow(test_x[idx].reshape((28, 28)))
# plt.show()
