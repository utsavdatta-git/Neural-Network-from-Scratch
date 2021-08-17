from neural_network.activation_functions import Activation
import numpy as np


class HiddenLayer:
    """
        A class to define a hidden layer of the neural network, its parameters and methods

        Methods
        -------
        forward_output(input_data, eps):
            Returns output of a single layer in forward propagation
        backward_output(delta):
            Returns output of a single layer in backward propagation
    """

    def __init__(self, n_in, n_out, activation_previous_layer='relu', activation='relu',
                 dropout=False, drop_prob=0.5, batch_norm=False):
        """Initializes all attributes of a hidden layer

        Attributes
        ----------
        n_in : int
            no. of incoming nodes of the layer
        n_out : int
            no. of outgoing nodes of the layer
        activation_previous_layer : str
            name of the activation function of the previous layer (required during backpropagation) (default relu)
        activation : str
            name of the activation function of the current layer (default relu)
        dropout : boolean
            whether to apply dropout or not (default False)
        drop_prob : float
            dropout probability (default 0.5)
        batch_norm : boolean
            whether to apply batch normalization or not (default False)
        """
        # Input initialization and attaching activations to the layer
        self.input = None
        self.output = None
        self.activation_name = activation
        self.activation = Activation(activation).activation_func
        self.activation_deriv = Activation(activation_previous_layer).activation_func_deriv

        # Initialization of weight and bias matrices
        self.W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
        self.b = np.zeros(n_out, )
        self.n_in = n_in
        self.n_out = n_out

        # Initialization of dropout properties
        self.dropout = dropout
        self.dropout_prob = drop_prob

        # Initialization of gradient matrices
        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)

        # Initialization of momentum parameters
        self.velocity_W = np.zeros(self.W.shape)
        self.velocity_b = np.zeros(self.b.shape)

        # Initialization fo batch normalization parameters
        self.batch_norm = batch_norm
        self.gamma_batch_norm = np.ones((1, n_in))
        self.beta_batch_norm = 0
        self.grad_gamma_batch_norm = np.ones(self.gamma_batch_norm.shape)
        self.grad_beta_batch_norm = 0

    def forward_output(self, input_data, is_training=True, eps=1e-8):
        """Returns output of a single layer in forward propagation

        Args
        ----------
        input_data : numpy array
            input data to the hidden layer
        eps : float
            epsilon parameter for numerical stability in batch normalization
        is_training : boolean
            indicator to whether running in training or prediction mode

        Returns
        ----------
        output : numpy array
            output of a single layer
        """
        self.input = input_data
        if self.dropout and is_training:
            keep_prob = 1 - self.dropout_prob
            prob = np.random.rand(input_data.shape[0], input_data.shape[1])
            prob = prob < keep_prob
            self.input = np.multiply(self.input, prob)
            self.input = self.input / keep_prob

        if self.batch_norm:
            input_mean = self.input.mean(axis=0, keepdims=True)
            input_var = self.input.var(axis=0, keepdims=True)
            self.input = (self.input - input_mean) / np.sqrt(input_var + eps)  # added eps to avoid divide by zero
            self.input = self.input * self.gamma_batch_norm + self.beta_batch_norm

        layer_linear_output = np.dot(self.input, self.W) + self.b
        self.output = layer_linear_output if self.activation_name is None \
            else self.activation(layer_linear_output)
        self.W = np.nan_to_num(self.W)
        return self.output

    def backward_output(self, delta):
        """Returns output of a single layer in backward propagation

        Args
        ----------
        delta : numpy array
            delta from previous layer in backpropagation

        Returns
        ----------
        layer_delta : numpy array
            delta of a single layer in backward propagation
        """
        layer_delta = None
        self.grad_W = np.atleast_2d(self.input).T.dot(np.atleast_2d(delta))
        self.grad_b = np.sum(delta)
        if self.activation_deriv:
            layer_delta = delta.dot(self.W.T) * self.activation_deriv(self.input)

            if self.batch_norm:
                self.grad_gamma_batch_norm = np.mean(layer_delta, axis=0, keepdims=True) * self.gamma_batch_norm
                self.grad_beta_batch_norm = np.mean(layer_delta)
                layer_delta = layer_delta * self.gamma_batch_norm
        return layer_delta