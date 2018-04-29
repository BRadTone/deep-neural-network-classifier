import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tests.mnist import test_x, test_y
from src.utils import predict
from tests.helpers import normalize

path = '../datasets/mnist/model-alpha-0.18-iterations-150.pickle'
params = {}
if os.path.isfile(path):
    params = pickle.load(open(path, 'rb'))

idx = 2

# x = normalize(test_x[idx]).reshape((784, 1))
# aL, _ = predict(x, params)
# print(np.argmax(aL))
# print(test_y[idx])


m_test = test_x.shape[0]
# todo: import mean and std from training
x = normalize(test_x.reshape((784, m_test)))
AL, _ = predict(x, params)

print(np.argmax(AL[0][idx]))
print(test_y[idx])
AL = np.argmax(AL, axis=0).reshape(1, m_test)

diff = AL == test_y

plt.imshow(test_x[idx].reshape((28, 28)))
plt.show()
