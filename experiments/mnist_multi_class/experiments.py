import numpy as np
import pandas as pd
from neural_network.mlp import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from neural_network.utilities.metrics import accuracy
import matplotlib.pyplot as plt


def data_transform_pipeline(in_train_data_orig, in_test_data_orig, in_test_data_submission):
    """Pre-processes and transforms input data

    Args
    ----------
    in_train_data_orig : pandas dataframe
        training data
    in_test_data_orig : pandas dataframe
         test_data
    in_test_data_submission : pandas dataframe
        test data to make final predictions for submission

    Returns
    ----------
    scaled_train_data : numpy array
        transformed training data
    scaled_test_data : numpy array
        transformed testing data
    one_hot_train_label : numpy array
        one hot encoded training target labels
    one_hot_test_label : numpy array
        one hot encoded testing target labels
    """
    train_data = in_train_data_orig.iloc[:, 1:]
    test_data = in_test_data_orig.iloc[:, 1:]
    test_data_submission = in_test_data_submission
    train_label = np.array(in_train_data_orig["label"])
    test_label = np.array(in_test_data_orig["label"])

    scaler = StandardScaler()
    scaler.fit(train_data)
    scaled_train_data = scaler.transform(train_data)
    scaled_test_data = scaler.transform(test_data)
    scaled_test_submission = scaler.transform(test_data_submission)

    train_label = train_label.reshape(-1, 1)
    test_label = test_label.reshape(-1, 1)
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(train_label)
    one_hot_train_label = enc.transform(train_label).toarray()
    one_hot_test_label = enc.transform(test_label).toarray()

    return scaled_train_data, scaled_test_data, one_hot_train_label, one_hot_test_label, scaled_test_submission


def generate_plots(in_baseline_training_history=None, in_compare_training_history=None, plot_title=""):
    """Plot model accuracy and loss figures

    Args
    ----------
    in_baseline_training_history : dictionary
        loss and accuracy log of model training of the baseline model
    in_compare_training_history : dictionary
        loss and accuracy log of model training of the model being compared
    plot_title : str
        title of the plot
    """
    if in_baseline_training_history:
        plt.plot(in_baseline_training_history['train_accuracy'])
        plt.plot(in_baseline_training_history['val_accuracy'])
        plt.title(f"{plot_title} accuracy")
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(f"{plot_title} accuracy.png")
        plt.clf()

        plt.plot(in_baseline_training_history['train_loss'])
        plt.plot(in_baseline_training_history['val_loss'])
        plt.title(f"{plot_title} loss")
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(f"{plot_title} loss.png")
        plt.clf()

    if in_baseline_training_history and in_compare_training_history:
        plt.plot(in_baseline_training_history['train_accuracy'])
        plt.plot(in_baseline_training_history['val_accuracy'])
        plt.plot(in_compare_training_history['train_accuracy'])
        plt.plot(in_compare_training_history['val_accuracy'])
        plt.title(f"{plot_title} accuracy")
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['baseline_train', 'baseline_val', 'new_train', 'new_val'], loc='upper left')
        plt.savefig(f"{plot_title} accuracy.png")
        plt.clf()

        plt.plot(in_baseline_training_history['train_loss'])
        plt.plot(in_baseline_training_history['val_loss'])
        plt.plot(in_compare_training_history['train_loss'])
        plt.plot(in_compare_training_history['val_loss'])
        plt.title(f"{plot_title} loss")
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['baseline_train', 'baseline_val', 'new_train', 'new_val'], loc='upper left')
        plt.savefig(f"{plot_title} loss.png")
        plt.clf()


def run_all_experiments():
    """Runs multiple experiments based on different combinations of hyper-parameters,
       calculates test accuracy and plots loss and accuracy metrics

    """
    train_data_orig = pd.read_csv("data/mnist_train.csv")
    test_data_orig = pd.read_csv("data/mnist_test.csv")
    test_data_orig_submission = pd.read_csv("data/mnist_test_for_submission.csv")

    X, X_test, y, y_test, X_test_submission = data_transform_pipeline(train_data_orig, test_data_orig,
                                                                      test_data_orig_submission)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, shuffle=True)

    mlp = MLPClassifier((784, 128, 10), (None, "relu", "softmax"))  # define a baseline model
    baseline_training_history = mlp.fit(X_train, y_train, X_val, y_val, epochs=50,
                                        optimizer="batch_gradient_descent",
                                        verbose=False)  # train baseline model
    y_test_pred = mlp.predict(X_test)
    print(f"baseline model test accuracy: {accuracy(y_test_pred, y_test)}")
    generate_plots(baseline_training_history, plot_title="baseline model")

    mlp2 = MLPClassifier((784, 128, 10), (None, "relu", "softmax"))
    training_history = mlp2.fit(X_train, y_train, X_val, y_val, epochs=50,
                                optimizer="mini_batch_gradient_descent",
                                verbose=False, mini_batch_size=128)

    y_test_pred = mlp2.predict(X_test)
    print(f"baseline+mini batch model test accuracy: {accuracy(y_test_pred, y_test)}")
    generate_plots(in_baseline_training_history=baseline_training_history, in_compare_training_history=training_history,
                   plot_title="baseline+mini_batch model")

    mlp3 = MLPClassifier((784, 128, 10), (None, "relu", "softmax"))
    training_history = mlp3.fit(X_train, y_train, X_val, y_val, epochs=50,
                                optimizer="mini_batch_gradient_descent",
                                verbose=False, mini_batch_size=128, momentum=True)
    y_test_pred = mlp3.predict(X_test)
    print(f"baseline+mini batch+momentum model test accuracy: {accuracy(y_test_pred, y_test)}")
    generate_plots(baseline_training_history, training_history, plot_title="baseline+mini_batch+momentum model")

    mlp4 = MLPClassifier((784, 128, 10), (None, "relu", "softmax"), dropout=True)
    training_history = mlp4.fit(X_train, y_train, X_val, y_val, epochs=50,
                                optimizer="mini_batch_gradient_descent",
                                verbose=False, mini_batch_size=128, momentum=True)
    y_test_pred = mlp4.predict(X_test)
    print(f"baseline+mini batch+momentum+dropout model test accuracy: {accuracy(y_test_pred, y_test)}")
    generate_plots(baseline_training_history, training_history, plot_title="baseline+mini_batch+momentum+dropout model")

    mlp5 = MLPClassifier((784, 128, 10), (None, "relu", "softmax"), batch_norm=True)
    training_history = mlp5.fit(X_train, y_train, X_val, y_val, epochs=50,
                                optimizer="mini_batch_gradient_descent",
                                verbose=False, mini_batch_size=128, momentum=True)
    y_test_pred = mlp5.predict(X_test)
    print(f"baseline+mini batch+momentum+batch_norm model test accuracy: {accuracy(y_test_pred, y_test)}")
    generate_plots(baseline_training_history, training_history,
                   plot_title="baseline+mini_batch+momentum+batch_norm model")

    # final experiment for submission
    mlp_final = MLPClassifier((784, 128, 128, 64, 10), (None, "tanh", "tanh", "tanh", "softmax"))
    training_history = mlp_final.fit(X_train, y_train, X_val, y_val, epochs=100,
                                     optimizer="mini_batch_gradient_descent",
                                     mini_batch_size=128, momentum=True)
    y_test_pred_probs = mlp_final.predict(X_test)
    print(f"final model test accuracy: {accuracy(y_test_pred_probs, y_test)}")
    generate_plots(training_history,
                   plot_title="final model model")
    y_pred_probs_final = mlp_final.predict(X_test_submission)
    y_pred_final = np.argmax(y_pred_probs_final, axis=1)

    # create final csv from the predictions for submission
    y_pred_df = pd.DataFrame(data=y_pred_final, columns=["Label"])
    y_pred_df['ImageId'] = y_pred_df.index
    y_pred_df.to_csv("data/submission.csv", index=False)


if __name__ == "__main__":
    run_all_experiments()
