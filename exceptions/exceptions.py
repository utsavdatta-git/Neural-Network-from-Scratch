class ModelDefinitionError(Exception):
    """Exception raised when every layer defined in the neural network model do not have
    corresponding activation functions.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


class UnknownActivationError(Exception):
    """Exception raised when an unknown activation function is chosen during model definition.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


class IncompatibleShapeErrorInput(Exception):
    """Exception raised when shapes of input layer and input to it are incompatible

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


class IncompatibleShapeErrorOutput(Exception):
    """Exception raised when shapes of output layer and no. of classes to predict are incompatible

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message
