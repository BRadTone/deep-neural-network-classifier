import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer

from src.init_params import init_params
from src.forward_prop import model_forward
from src.back_prop import model_back_prop
from src.utils import cost_fn, update_params


# todo: decorator for time benchmark, decorator for plotting when iterating
def model(X, Y, X_test, Y_test, layers_dims, learning_rate=0.0075, epochs=3000, weight_scale=0.01, print_cost=False):
    """
    Arguments:
    X -- training_data
    Y -- true "label" vector

    layers_dims  -- hyperparameter -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate  -- hyperparameter -- learning rate of the gradient descent update rule
    num_iterations -- hyperparameter --  number of iterations of the optimization loop
    weight_scale -- hyperparameter -- to control  values of weight matrices

    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    costs = []
    costs_test = []
    params = init_params(layers_dims, weight_scale)

    plt.ion()
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.draw()

    # Loop (gradient descent)
    start = timer()
    for i in range(0, epochs):
        AL, caches = model_forward(X, params)

        cost = cost_fn(AL, Y)

        grads = model_back_prop(AL, Y, caches)

        params = update_params(params, grads, learning_rate)

        # Print and plot cost
        if print_cost and i % 2 == 0:
            AL_test, _ = model_forward(X_test, params)
            cost_test = cost_fn(AL_test, Y_test)

            end = timer()
            print("Cost after iteration %i: %f" % (i, cost), 'and it took: ', round(end - start, 2), 's')
            costs.append(cost)
            costs_test.append(cost_test)

            plt.plot(np.squeeze(costs), label='train')
            plt.plot(np.squeeze(costs_test), label='test')
            plt.draw()
            plt.pause(0.1)
            plt.clf()

    return params, (cost, cost_test)
