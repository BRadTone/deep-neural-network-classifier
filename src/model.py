import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer

from src.init_params import init_params
from src.forward_prop import model_forward
from src.back_prop import model_back_prop
from src.utils import cost_fn, update_params

plt.rcParams.update({'font.size': 7})
plt.style.use('Solarize_Light2')


# todo: decorator for time benchmark, decorator for plotting when iterating
def model(X, Y, X_valid, Y_valid, layers_dims, learning_rate=0.0075, epochs=3000, weight_scale=0.01, print_cost=False):
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
    costs_valid = []
    params = init_params(layers_dims, weight_scale)

    plt.ion()

    plt.ylabel('cost')
    plt.xlabel('iterations (per 2)')
    plt.title("Learning rate =" + str(learning_rate))

    costs_line, = plt.plot(costs, [], 'r', label='training set')
    costs_valid_line, = plt.plot(costs_valid, [], 'b--', label='cross validation set')

    ax = plt.gca()
    ax.grid()
    ax.legend()
    # Loop (gradient descent)
    start = timer()
    for i in range(0, epochs):
        AL, caches = model_forward(X, params)

        cost = cost_fn(AL, Y)

        grads = model_back_prop(AL, Y, caches)

        params = update_params(params, grads, learning_rate)

        # Print and plot cost
        if print_cost and i % 1 == 0:
            end = timer()
            AL_test, _ = model_forward(X_valid, params)

            print('Cost after iteration {0}: {1}, after {2}s.'.format(i, round(cost, 4), round(end - start, 1)))
            costs.append(cost)
            costs_valid.append(cost_fn(AL_test, Y_valid))

            range_ = np.arange(len(costs))
            costs_line.set_data(range_, costs)
            costs_valid_line.set_data(range_, costs_valid)

            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(.1)

    return params, (costs, costs_valid)
