import numpy as np


class Initializers:
    @classmethod
    def weighted_uniform(cls, layers_dims, weight=0.01):
        return cls.base(layers_dims, factor=lambda: weight)

    @classmethod
    def weighted_normal(cls, layers_dims, weight=0.01):
        return cls.base(layers_dims, generator=np.random.randn, factor=lambda: weight)

    @classmethod
    def xavier(cls, layers_dims):
        def factor(l):
            return np.sqrt(2 / (layers_dims[l - 1] + layers_dims[l]))

        return cls.base(layers_dims, generator=np.random.randn, factor=factor)

    @classmethod
    def he_initializaion(cls, layers_dims):
        def factor(l):
            return np.sqrt(2 / layers_dims[l - 1])

        return cls.base(layers_dims, generator=np.random.randn, factor=factor)

    @classmethod
    def base(cls, layers_dims, generator=np.random.rand, factor=lambda: 1.):
        parameters = {}
        L = len(layers_dims)  # number of layers in the network

        for l in range(1, L):
            parameters['W' + str(l)] = generator(layers_dims[l], layers_dims[l - 1]) * factor(l)
            parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

            assert (parameters['W' + str(l)].shape == (layers_dims[l], layers_dims[l - 1]))
            assert (parameters['b' + str(l)].shape == (layers_dims[l], 1))

        return parameters
