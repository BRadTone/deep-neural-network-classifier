from src.init_params import init_params
from src.forward_prop import model_forward
from src.back_prop import model_back_prop
from src.utils import cost_fn, update_params


def model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, weight_scale=0.01, print_cost=False):
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

    params = init_params(layers_dims, weight_scale)

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        AL, caches = model_forward(X, params)

        cost = cost_fn(AL, Y)

        grads = model_back_prop(AL, Y, caches)

        params = update_params(params, grads, learning_rate)

        # Print the cost every 100 training example
        if print_cost and i % 5 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
            costs.append(cost)


    # # plot the cost
    # plt.plot(np.squeeze(costs))
    # plt.ylabel('cost')
    # plt.xlabel('iterations (per tens)')
    # plt.title("Learning rate =" + str(learning_rate))
    # plt.show()
    return params

