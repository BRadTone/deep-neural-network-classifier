import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tests.mnist import test_x, test_y
from src.utils import predict
from tests.helpers import normalize

from src.utils import cost_fn

path = '../datasets/mnist/model-alpha-0.18-iterations-2500-layers-3.pickle'
params = {}
if os.path.isfile(path):
    params = pickle.load(open(path, 'rb'))

idx = 0

# x = normalize(test_x[idx]).reshape((784, 1))
# aL, _ = predict(x, params)
# print(np.argmax(aL))
# print(test_y[idx])
m_test = test_x.shape[0]

test_Y_v = np.zeros((10, m_test))
test_Y_v[test_y.reshape(1, m_test), np.arange(m_test)] = 1

# todo: import mean and std from training
x = normalize(test_x.reshape((784, m_test)))
AL, _ = predict(x, params)
cost = cost_fn(AL, test_Y_v)
# print(np.argmax(AL[0][0]))
AL = np.argmax(AL, axis=0).reshape(1, m_test)
test_y = test_y.reshape(1, m_test)
diff = AL == test_y


print(cost)
# print(np.sum(diff))
# plt.imshow(test_x[idx].reshape((28, 28)))
# plt.show()
