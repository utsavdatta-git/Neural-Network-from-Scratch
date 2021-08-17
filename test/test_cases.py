import tensorflow as tf
from neural_network.utilities.losses import cross_entropy
import numpy as np
from neural_network.activation_functions import Activation
from neural_network.mlp import MLPClassifier


def test_loss():
    y_true = np.array([[0, 1, 0], [0, 0, 1]])
    y_pred = np.array([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
    cce = tf.keras.losses.CategoricalCrossentropy()
    cce_loss_keras = round(cce(y_true, y_pred).numpy(), 3)
    cce_loss_vanilla, _ = cross_entropy(y_true, y_pred, 2)
    if cce_loss_keras == round(cce_loss_vanilla, 3):
        print("Loss test passed")


def test_relu():
    relu_layer_keras = tf.keras.layers.ReLU()
    relu_output_keras = relu_layer_keras([-3.0, -1.0, 0.0, 2.0]).numpy()
    relu_layer_vanilla = Activation(activation_func_name="relu")
    relu_output_vanilla = relu_layer_vanilla.activation_func(np.array([-3.0, -1.0, 0.0, 2.0]))
    if np.isclose(relu_output_keras, relu_output_vanilla, rtol=1e-05, atol=1e-08, equal_nan=False).all():
        print("Relu test passed")


def test_tanh():
    tanh_output_keras = np.around(tf.keras.activations.tanh([-3.0, -1.0, 0.0, 2.0]).numpy(), 3)
    tanh_layer_vanilla = Activation(activation_func_name="tanh")
    tanh_output_vanilla = np.around(tanh_layer_vanilla.activation_func(np.array([-3.0, -1.0, 0.0, 2.0])), 3)
    if np.isclose(tanh_output_keras, tanh_output_vanilla, rtol=1e-05, atol=1e-08, equal_nan=False).all():
        print("Tanh test passed")


def test_softmax():
    inputs = tf.random.normal(shape=(4, 4))
    smax_output_keras = np.around(tf.keras.activations.softmax(inputs).numpy(), 3)
    smax_layer_vanilla = Activation(activation_func_name="softmax")
    smax_output_vanilla = np.around(smax_layer_vanilla.activation_func(np.array(inputs)), 3)
    if np.isclose(smax_output_keras, smax_output_vanilla, rtol=1e-05, atol=1e-08, equal_nan=False).all():
        print("Softmax test passed")


def test_cce_grad():
    inputs = np.random.normal(size=(4, 5))
    mlp = MLPClassifier((5, 4, 3), (None, "relu", "softmax"))  # create the vanilla model

    model = tf.keras.models.Sequential()  # create the keras model
    model.add(tf.keras.layers.InputLayer(5))
    model.add(tf.keras.layers.Dense(4, activation='relu'))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))
    model.layers[0].set_weights(
        [mlp.layers[0].W, mlp.layers[0].b])  # make initial weights of keras model = vanilla model initial weights
    model.layers[0].set_weights([mlp.layers[0].W, mlp.layers[0].b])
    model.layers[1].set_weights([mlp.layers[1].W, mlp.layers[1].b])
    model.layers[1].set_weights([mlp.layers[1].W, mlp.layers[1].b])
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(inputs, np.array([[0, 1, 0], [0, 0, 0], [0, 1, 1], [1, 0, 1]]), batch_size=1, epochs=5, shuffle=False)

    mlp.fit(inputs, np.array([[0, 1, 0], [0, 0, 0], [0, 1, 1], [1, 0, 1]]),
            inputs, np.array([[0, 1, 0], [0, 0, 0], [0, 1, 1], [1, 0, 1]]), learning_rate=0.01,
            epochs=5, mini_batch_size=1)
    print("<================== Vanilla model weights and biases after 1 iteration ==================>")
    print(f"Weights: {mlp.layers[1].W}, \nBiases: {mlp.layers[1].b}")
    print("<================== Keras model weights and biases after 1 iteration ==================>")
    print(f"Weights: {model.layers[1].get_weights()}")


if __name__ == "__main__":
    test_loss()
    test_relu()
    test_tanh()
    test_softmax()
    test_cce_grad()
