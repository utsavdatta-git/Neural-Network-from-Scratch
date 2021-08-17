from neural_network.hidden_layer import HiddenLayer
from neural_network.utilities.losses import cross_entropy
from neural_network.utilities.metrics import accuracy
from exceptions.exceptions import ModelDefinitionError, UnknownActivationError, IncompatibleShapeErrorInput, \
    IncompatibleShapeErrorOutput
import numpy as np


def _prepare_random_mini_batches(x, y, mini_batch_size):
    """Implements forward propagation through the network

    Args
    ----------
    x : numpy array
        input data to create mini batches from
    y : numpy array
        input labels to create mini batches from
    mini_batch_size : int
        size of mini batches
    Returns
    ----------
    mini_batches : list
        list of mini batches
    """
    num_samples = x.shape[0]
    mini_batches = []
    permutation = list(np.random.permutation(num_samples))
    rand_x = x[permutation, :]
    rand_y = y[permutation, :]

    num_mini_batches = num_samples // mini_batch_size
    for i in range(num_mini_batches):
        mini_batch_x = rand_x[i * mini_batch_size: (i + 1) * mini_batch_size, :]
        mini_batch_y = rand_y[i * mini_batch_size: (i + 1) * mini_batch_size, :]
        mini_batch = (mini_batch_x, mini_batch_y)
        mini_batches.append(mini_batch)

    if num_samples % mini_batch_size != 0:
        mini_batch_x = rand_x[num_mini_batches * mini_batch_size:, :]
        mini_batch_y = rand_y[num_mini_batches * mini_batch_size:, :]

        mini_batch = (mini_batch_x, mini_batch_y)
        mini_batches.append(mini_batch)
    return mini_batches


class MLPClassifier:
    """A class to define the neural network model

    Methods
    -------
    forward_propagation(input_data, is_training):
        Executes a single run of forward propagation through the network
    backpropagation(delta):
        Executes a single run of backward propagation through the network
    update(learning_rate, weight_decay, weight_decay_param, momentum, batch_norm):
        Updates weights and biases of the network
    predict(x, is_training):
        Makes predictions based on input data
    fit(x_train, y_train, x_val, y_val, learning_rate, epochs,
            optimizer, mini_batch_size, weight_decay, weight_decay_param, momentum):
        Performs model training using batch gradient descent or mini batch gradient descent
    """

    def __init__(self, hidden_layer_sizes=(64, 32, 32, 10), activation=(None, 'relu', 'relu', 'softmax'),
                 dropout=False, drop_prob=0.5, batch_norm=False):
        """Initializes attributes of the neural network model

        Args
        ----------
        hidden_layer_sizes : tuple
            integer tuple of no. of neuron for each layer
            (default (64, 32, 32, 10))
        activation : tuple
            string tuple of activation function of each layer
            (default (None, 'relu', 'relu', 'softmax'))
        dropout : boolean
             whether to apply dropout or not (default False)
        drop_prob : float
            dropout probability (default 0.5)
        batch_norm : boolean
            whether to apply batch normalization or not (default False)

        Raises
        ------
        ModelDefinitionError : if each layer is not associated with an activation function
        UnknownActivationError :  if an unknown activation function is chosen other than
                                  None, 'relu', 'softmax' or 'tanh'
        """

        if len(hidden_layer_sizes) != len(activation):
            raise (ModelDefinitionError("Each layer should have a corresponding activation function defined, \n"
                                        "please mention 'None' if no activation is associated with a layer."))

        for activation_function in set(activation):
            if activation_function not in (None, 'relu', 'softmax', 'tanh'):
                raise (UnknownActivationError("Unknown activation function chosen, \n"
                                              "please choose from (None, 'relu', 'softmax', 'tanh')"))

        self.layers = []
        self.params = []
        self.activation = activation
        self.batch_norm = batch_norm

        for i in range(len(hidden_layer_sizes) - 1):
            self.layers.append(HiddenLayer(
                hidden_layer_sizes[i], hidden_layer_sizes[i + 1], activation[i], activation[i + 1],
                dropout=dropout, drop_prob=drop_prob, batch_norm=batch_norm))

    def _forward_propagation(self, input_data, is_training=True):
        """Implements forward propagation through the network

        Args
        ----------
        input_data : numpy array
            input data to the hidden layer
        is_training : boolean
            whether called during training or testing (default True)

        Returns
        ----------
        output : numpy array
            output of the forward propagation
        """
        output = None
        for layer in self.layers:
            output = layer.forward_output(input_data, is_training)
            input_data = output
        return output

    def _backpropagation(self, delta):
        """Implements backward propagation through the network

        Args
        ----------
        delta : numpy array
            input delta from last layer (error from loss function)

        Returns
        ----------
        None
        """
        accumulated_delta = self.layers[-1].backward_output(delta)
        for layer in reversed(self.layers[:-1]):
            accumulated_delta = layer.backward_output(accumulated_delta)

    def _update(self, learning_rate, weight_decay, weight_decay_param, momentum, batch_norm):
        """Updates weights and biases based on gradients

        Args
        ----------
        learning_rate : float
            learning rate parameter
        weight_decay : boolean
            whether to apply weight decay
        weight_decay_param : float
            weight decay parameter
        momentum : boolean
            whether to apply momentum
        batch_norm : boolean
            whether to apply batch normalization
        Returns
        ----------
        None
        """
        if batch_norm or momentum or weight_decay:
            if batch_norm:
                for layer in self.layers:
                    layer.W -= learning_rate * layer.grad_W
                    layer.b -= learning_rate * layer.grad_b
                    layer.gamma_batch_norm -= learning_rate * layer.grad_gamma_batch_norm
                    layer.beta_batch_norm -= learning_rate * layer.grad_beta_batch_norm

            if momentum:
                for _, layer in enumerate(self.layers):
                    layer.velocity_W = 0.001 * layer.velocity_W + learning_rate * layer.grad_W
                    layer.W -= layer.velocity_W
                    layer.velocity_b = 0.001 * layer.velocity_b + learning_rate * layer.grad_b
                    layer.b -= layer.velocity_b

            if weight_decay:
                for layer in self.layers:
                    layer.W = (1 - learning_rate * weight_decay_param) * layer.W - (learning_rate * layer.grad_W)
                    layer.b -= learning_rate * layer.grad_b

        else:
            for layer in self.layers:
                layer.W -= learning_rate * layer.grad_W
                layer.b -= learning_rate * layer.grad_b

    def predict(self, x, is_training=False):
        """Makes predictions on input data

        Args
        ----------
        x : numpy array
            input data
        is_training : boolean
             whether called during training or testing (default False)
        Returns
        ----------
        output : numpy array
            predictions
        """
        x = np.array(x)
        output = self._forward_propagation(x, is_training)
        return output

    def fit(self, x_train, y_train, x_val, y_val, learning_rate=0.01, epochs=100,
            optimizer='mini_batch_gradient_descent', mini_batch_size=64, weight_decay=False,
            weight_decay_param=0.01, momentum=False, verbose=True):
        """Performs model training using batch gradient descent or mini batch gradient descent

        Args
        ----------
        x_train : numpy array
            input training data
        y_train : numpy array
            input training labels
        x_val : numpy array
            input validation data
        y_val : numpy array
            input validation labels
        learning_rate : float
            learning rate parameter (default 0.01)
        epochs : int
            no. of epochs for training (default 100)
        optimizer : str
            optimizer to be used (default mini_batch_gradient_descent)
        mini_batch_size : int
            size of each mini batch, valid only if using mini_batch_gradient_descent (default 64)
        weight_decay : boolean
            whether to use weight decay (default False)
        weight_decay_param : float
            parameter for weight decay (default 0.01)
        momentum : boolean
            whether to apply momentum (default False)
        verbose : boolean
            whether to show verbose messages of the training process

        Returns
        ----------
        history_dict : dict of lists
            dictionary of lists containing training, validation loss and accuracy

        Raises
        ------
        IncompatibleShapeErrorInput : raised when shape of input layer and input to it are incompatible
        IncompatibleShapeErrorOutput : raised when shapes of output layer and no. of classes to predict are incompatible
        """
        # some sanity checks before beginning the training
        if x_train.shape[1] != self.layers[0].n_in or x_val.shape[1] != self.layers[0].n_in:
            raise (IncompatibleShapeErrorInput("Shape of input data and no. of neurons "
                                               "in the input layer are incompatible."))

        if y_train.shape[1] != self.layers[-1].n_out or y_val.shape[1] != self.layers[-1].n_out:
            raise (IncompatibleShapeErrorOutput("No. of neurons in the output layer and no. of "
                                                "classes to predict are incompatible."))

        history_dict = {"train_loss":[], "train_accuracy":[],
                        "val_loss":[], "val_accuracy":[]} # define dictionary for tracking training metrics
        if optimizer == 'batch_gradient_descent':
            for epoch in range(epochs):
                train_y_pred = self._forward_propagation(x_train)
                train_loss, delta = cross_entropy(y_train, train_y_pred, x_train.shape[0])
                history_dict["train_loss"].append(train_loss)
                self._backpropagation(delta)
                self._update(learning_rate, weight_decay, weight_decay_param, momentum, self.batch_norm)
                train_output = np.zeros_like(train_y_pred)
                train_output[np.arange(len(train_y_pred)), train_y_pred.argmax(1)] = 1
                train_acc = accuracy(train_output, y_train)
                history_dict["train_accuracy"].append(train_acc)
                val_y_pred = self.predict(x_val)
                val_loss, _ = cross_entropy(y_val, val_y_pred, x_val.shape[0])
                history_dict["val_loss"].append(val_loss)
                val_output = np.zeros_like(val_y_pred)
                val_output[np.arange(len(val_y_pred)), val_y_pred.argmax(1)] = 1
                val_acc = accuracy(val_output, y_val)
                history_dict["val_accuracy"].append(val_acc)

                if verbose:
                    print(f"epoch: {str(epoch + 1)} =======> train_loss: {round(train_loss, 5)}, "
                          f"train_accuracy: {round(train_acc, 5)}, "
                          f"val_loss: {round(val_loss, 5)}, val_accuracy: {round(val_acc, 5)}")

        if optimizer == 'mini_batch_gradient_descent':
            for epoch in range(epochs):
                train_loss = 0
                train_mini_batches = _prepare_random_mini_batches(x_train, y_train, mini_batch_size)
                for train_mini_batch_no, train_mini_batch in enumerate(train_mini_batches):
                    (train_mini_batch_x, train_mini_batch_y) = train_mini_batch
                    train_mini_batch_y_pred = self._forward_propagation(train_mini_batch_x)
                    train_mini_batch_loss, delta = cross_entropy(train_mini_batch_y, train_mini_batch_y_pred,
                                                                 mini_batch_size)
                    train_loss += train_mini_batch_loss
                    self._backpropagation(delta)
                    self._update(learning_rate, weight_decay, weight_decay_param, momentum, self.batch_norm)

                train_y_pred = self.predict(x_train)
                train_output = np.zeros_like(train_y_pred)
                train_output[np.arange(len(train_y_pred)), train_y_pred.argmax(1)] = 1
                train_acc = accuracy(train_output, y_train)

                val_y_pred = self.predict(x_val)
                val_loss, _ = cross_entropy(y_val, val_y_pred, x_val.shape[0])
                val_output = np.zeros_like(val_y_pred)
                val_output[np.arange(len(val_y_pred)), val_y_pred.argmax(1)] = 1
                val_acc = accuracy(val_output, y_val)

                history_dict["train_accuracy"].append(train_acc)
                history_dict["val_accuracy"].append(val_acc)
                history_dict["train_loss"].append(train_loss / mini_batch_size)
                history_dict["val_loss"].append(val_loss)

                if verbose:
                    print(f"epoch: {str(epoch + 1)} =======> train_loss: {round(train_loss / mini_batch_size, 5)}, "
                          f"train_accuracy: {round(train_acc, 5)}, "
                          f"val_loss: {round(val_loss, 5)}, val_accuracy: {round(val_acc, 5)}")
        return history_dict