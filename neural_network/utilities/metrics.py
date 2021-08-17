import numpy as np


def accuracy(predicted, ground_truth):
    """Calculates accuracy metric

    Args
    ----------
    predicted : numpy array
        predicted probabilities
    ground_truth : numpy array
         ground truth one hot encoded labels

    Returns
    ----------
    accuracy : float
        accuracy value
    """
    predicted_labels_decoded = np.argmax(predicted, axis=1)
    ground_truth_labels_decoded = np.argmax(ground_truth, axis=1)
    correct_rate = [1 if pred == truth else 0 for (pred, truth) in
                    zip(predicted_labels_decoded, ground_truth_labels_decoded)]
    accuracy = sum(correct_rate) / ground_truth_labels_decoded.size
    return accuracy * 100
