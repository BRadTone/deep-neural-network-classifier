import glob
import os
import gzip
import pickle
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from src.model import model

with gzip.open('../datasets/mnist/mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

train_x, train_y = train_set
m = train_x.shape[0]
# plt.imshow(train_x[9].reshape((28, 28)), cmap=cm.Greys_r)
# print(train_y)
# plt.show()


model_shape = [28 * 28, 10, 10]

y = np.zeros((10, m))
y[train_y.reshape(1, m) - 1, np.arange(m)] = 1

hyp_params = {
    'iters': 100,
    'alpha': 0.9
}

model = model(train_x.T, y,
              layers_dims=model_shape,
              print_cost=True,
              num_iterations=hyp_params['iters'],
              learning_rate=hyp_params['alpha'])

# todo: convert to 'with' statement, not woking atm
pickle_name = '../datasets/mnist/model_alpha-%a-iterations-%i.pickle' % (hyp_params['alpha'], hyp_params['iters'])
pickle_out = open(pickle_name, 'wb')
pickle.dump(model, pickle_out)
pickle_out.close()

