import numpy as np


class ActivationFns:
    def __init__(self):
        pass

    @staticmethod
    def sigmoid(Z):
        A = 1 / (1 + np.exp(-Z))

        print('type of A',type(A))
        return A

    @staticmethod
    def relu(Z):
        A = np.maximum(0, Z)
        return A

    @staticmethod
    def leaky_relu(Z, slope=0.01):
        A = np.maximum(Z * slope, Z)

        return A

    @staticmethod
    def tanh(Z):
        A = np.tanh(Z)

        return A


class dActivationFns:
    def __init__(self):
        pass

    @staticmethod
    def sigmoid(dA, Z):
        s = 1 / (1 + np.exp(-Z))
        dZ = dA * s * (1 - s)

        assert (dZ.shape == Z.shape)

        return dZ

    @staticmethod
    def relu(dA, Z):
        dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
        dZ[Z <= 0] = 0

        assert (dZ.shape == Z.shape)

        return dZ

    @staticmethod
    def leaky_relu(dA, Z, slope=0.01):
        dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
        dZ[Z <= 0] = slope

        assert (dZ.shape == Z.shape)

        return dZ

    @staticmethod
    def tanh(dA, Z):
        dZ = 1 / np.power(np.cosh(dA), 2)
        return dZ


