import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer

from src.Initializers import Initializers
from src.ForwardProp import ForwardProp
from src.BackProp import BackProp

from src_old_functional.utils import cost_fn, update_params

plt.rcParams.update({'font.size': 7})
plt.style.use('Solarize_Light2')


# todo: decorator for time benchmark, decorator for plotting when iterating
def model(X, Y, X_valid, Y_valid, layers_dims, learning_rate=0.0075, epochs=3000, print_cost=False):
    costs = []
    costs_valid = []

    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))

    costs_line, = plt.plot(costs, [], 'r', label='training set')
    costs_valid_line, = plt.plot(costs_valid, [], 'b--', label='cross validation set')

    ax = plt.gca()
    ax.grid()
    ax.legend()

    params = Initializers.he(layers_dims)
    # Loop (gradient descent)
    start = timer()
    for i in range(0, epochs):
        AL, caches = ForwardProp.model_forward(X, params)

        cost = cost_fn(AL, Y)

        grads = BackProp.model_back_prop(AL, Y, caches)

        params = update_params(params, grads, learning_rate)

        # Print and plot cost
        if print_cost:
            end = timer()
            AL_valid, _ = ForwardProp.model_forward(X_valid, params)

            print('Cost after iteration {}: {}, after {}s.'.format(i, round(cost, 4), round(end - start, 1)))
            costs.append(cost)
            costs_valid.append(cost_fn(AL_valid, Y_valid))
            # todo: update plot heading with current costs and iteration nume
            range_ = np.arange(len(costs))
            costs_line.set_data(range_, costs)
            costs_valid_line.set_data(range_, costs_valid)

            ax.relim()
            ax.autoscale_view()
            plt.pause(.001)
            plt.draw()

    return params, (costs, costs_valid)
