import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 7})
plt.style.use('Solarize_Light2')
ax = plt.gca()


class LearningCurves:
    def __init__(self, learning_rate):
        self.ax = ax
        self.learning_rate = learning_rate
        self.costs_line, = plt.plot([], [], 'r', label='training set')
        self.costs_valid_line, = plt.plot([], [], 'b--', label='cross validation set')

    def init_plot(self, plot_every):
        self.ax.legend()
        self.ax.grid()
        plt.ylabel('cost')
        plt.xlabel('iterations (per {})'.format(plot_every))
        plt.title("Learning rate =" + str(self.learning_rate))

    def update_plot(self, costs_train, i, costs_valid=[]):
        print('Cost after iteration {}: {}'.format(i, round(costs_train[-1], 4)))
        # todo: update plot heading with current costs and iteration number
        self.costs_line.set_data(np.arange(len(costs_train)), costs_train)
        self.costs_valid_line.set_data(np.arange(len(costs_valid)), costs_valid)

        self.ax.relim()
        self.ax.autoscale_view()
        plt.pause(.001)
        plt.draw()
