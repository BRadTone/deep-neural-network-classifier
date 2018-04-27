import gzip
import pickle
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from src.model import model

with gzip.open('../datasets/mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

train_x, train_y = train_set

# plt.imshow(train_x[9].reshape((28, 28)), cmap=cm.Greys_r)
# print(train_y)
# plt.show()

# print(train_x.shape)
# print(type(train_x))

print(train_y.shape)
model_shape = [28 * 28, 10, 10, 10]

model(train_x.T, train_y, layers_dims=model_shape)
