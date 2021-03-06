import numpy as np
from src.ActivationUtils import ActivationFnsDerivatives


class BackProp:
    ActivationFnsDerivatives = ActivationFnsDerivatives
    
    @staticmethod
    def linear_backward(dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]
    
        dW = 1/m * np.dot(dZ, A_prev.T)
        db = 1/m * np.sum(dZ, keepdims=True, axis=1)
        dA_prev = np.dot(W.T, dZ)
    
        return dA_prev, dW, db
    
    @classmethod
    def linear_activation_backward(cls, dA, cache, activation):
        '''
        Arguments:
        dA -- post-activation gradient for current layer l
        cache -- tuple of values (linear_cache, activation_cache)
        activation -- the activation fn to be used in this layer
    
        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1)
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        '''
    
        assert (hasattr(activation, '__call__'))
    
        linear_cache, Z = cache
    
        dZ = activation(dA, Z)
        dA_prev, dW, db = cls.linear_backward(dZ, linear_cache)
    
        return dA_prev, dW, db
    
    @classmethod
    def model_back_prop(cls, AL, Y, caches):
        '''
        Arguments:
        AL -- probability vector, output of the forward propagation
        Y -- true 'label' vector
        caches -- list of caches
    
        Returns:
        grads -- A dictionary with the gradients
    
        '''
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)
    
        # Initializing the backpropagation
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
        current_cache = caches[len(caches) -1]
    
        grads['dA' + str(L-1)],\
        grads['dW' + str(L)],\
        grads['db' + str(L)] = cls.linear_activation_backward(dAL, current_cache, ActivationFnsDerivatives.sigmoid)
    
        # Loop from l=L-2 to l=0
        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev_temp, \
            dW_temp, \
            db_temp = cls.linear_activation_backward(grads['dA' + str(l + 1)], current_cache, cls.ActivationFnsDerivatives.relu)

            grads['dA' + str(l)] = dA_prev_temp
            grads['dW' + str(l + 1)] = dW_temp
            grads['db' + str(l + 1)] = db_temp
    
        return grads
    
