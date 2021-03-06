import os
import sys
import pickle
import numpy as np
from src.Initializers import Initializers
from src.ForwardProp import ForwardProp
from src.BackProp import BackProp
from src.helpers.learning_curves import LearningCurves


class Model:
    ForwardProp = ForwardProp
    BackProp = BackProp

    def __init__(self, layers_dims, initializer=Initializers.he):
        self.params = []
        self.layers_dims = layers_dims
        self.LearningCurves = None

        if layers_dims:
            self.init_params(initializer)

    def predict(self, X):
        """
        :return: probability distribution
        """
        AL, _ = self.ForwardProp.model_forward(X, self.params)
        AL_exps = np.exp(AL)

        return AL_exps / np.sum(AL_exps, axis=0)

    def train(self, X, Y,  X_valid=[], Y_valid=[], epochs=2000, learning_rate=0.001, print_cost=False):
        plot_every = 1
        self.LearningCurves = LearningCurves(learning_rate)
        self.LearningCurves.init_plot(plot_every)

        costs_train = []
        costs_valid = []
        for i in range(0, epochs):
            AL, caches = ForwardProp.model_forward(X, self.params)

            cost = self.cost(AL, Y)

            grads = self.BackProp.model_back_prop(AL, Y, caches)

            self.update_params(grads, learning_rate)

            # todo: allow X_valid,
            # todo: save cost/cost_valid
            if print_cost and i % plot_every == 0:
                if len(X_valid) and len(Y_valid):
                    AL_valid, _ = self.ForwardProp.model_forward(X_valid, self.params)
                    cost_valid = self.cost(AL_valid, Y_valid)
                    costs_valid.append(cost_valid)

                costs_train.append(cost)
                self.LearningCurves.update_plot(costs_train, i, costs_valid)

    def update_params(self, grads, learning_rate):
        L = len(self.params) // 2

        for l in range(L):
            self.params["W" + str(l + 1)] -= learning_rate * grads["dW" + str(l + 1)]
            self.params["b" + str(l + 1)] -= learning_rate * grads["db" + str(l + 1)]

    def forward_prop(self, X):
        return self.ForwardProp.model_forward(X, self.params)

    @staticmethod
    def cost(AL, Y):
        """
        Arguments:
        AL -- probability vector corresponding to your label predictions
        Y -- true "label" vector

        Returns:
        cost -- cross-entropy cost
        """
        assert AL.shape == Y.shape

        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
        cost = np.squeeze(cost)

        return cost

    def back_prop(self, AL, Y, caches):
        return self.BackProp.model_back_prop(AL, Y, caches)

    # params CRUD
    def load_params(self, path):
        try:
            with open(path, "rb") as f:
                self.params = pickle.load(f)
                self.update_layers_dims()
        except OSError as err:
            print(err)
            sys.exit()

    def save_params(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.params, f)

    def init_params(self, initializer):
        if initializer:
            self.params = initializer(self.layers_dims)
            return

        self.params = initializer(self.layers_dims)

    # params CRUD - end

    def update_layers_dims(self):
        # init new layer dims with first layer
        self.layers_dims = [self.params['W' + str(1)].shape[1]]

        for i in range(1, len(self.params) // 2 + 1):
            layer_units = self.params['W' + str(i)].shape[0]
            self.layers_dims.append(layer_units)
        assert len(self.layers_dims) - 1 == len(self.params) // 2
