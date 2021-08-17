import numpy as np


class Activation:
    """A class to define all activation functions and their derivatives

    Methods
    -------
    _relu(x):
        Applies relu function to input and returns the result
    _relu_deriv(a):
        Returns the relu derivative of the input
    _tanh(x):
        Applies tanh function to input and returns the result
    _tanh_deriv(a):
        Returns the tanh derivative of the input
     _softmax(x):
        Applies softmax function to input and returns the result
    _softmax_deriv(a):
        Returns the softmax derivative of the input
    """

    def _relu(self, x):
        i = (x < 0)
        x[i] = 0
        return x

    def _relu_deriv(self, a):
        if a.any() >= 0:
            return 1.0
        else:
            return 0.0

    def _tanh(self, x):
        return np.tanh(x)

    def _tanh_deriv(self, a):
        return 1 - (np.tanh(a)) ** 2

    def _softmax(self, x):
        x_max = x.max()
        x_norm = x - x_max
        return np.exp(x_norm) / np.expand_dims(np.exp(x_norm).sum(axis=1), axis=1)

    def _softmax_deriv(self, a):
        return a * (1 - a)

    def __init__(self, activation_func_name='relu'):
        """Initializes the definition of the activation function and its derivative

        Args
        ----------
            activation_func_name : str
                name of the activation function (softmax, relu or tanh)
        """
        if activation_func_name == 'softmax':
            self.activation_func = self._softmax
            self.activation_func_deriv = self._softmax_deriv
        elif activation_func_name == 'relu':
            self.activation_func = self._relu
            self.activation_func_deriv = self._relu_deriv
        elif activation_func_name == 'tanh':
            self.activation_func = self._tanh
            self.activation_func_deriv = self._tanh_deriv
        else:
            self.activation_func = None
            self.activation_func_deriv = None
