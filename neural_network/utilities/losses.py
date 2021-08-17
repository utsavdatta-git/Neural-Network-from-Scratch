import numpy as np


def cross_entropy(y_true, y_pred, no_of_samples):
    """Calculates and return cross entropy loss

    Args
    ----------
    y_true : numpy array
        ground truth labels
    y_pred : numpy array
         predicted labels
    no_of_samples : int
         no. of samples in the data

    Returns
    ----------
    loss : numpy array
        loss value
    error : numpy array
        error value
    """
    loss = (-1 / no_of_samples) * np.sum(y_true * (np.log(y_pred + 1e-15)))
    error = (y_pred - y_true) / no_of_samples
    return loss, error
