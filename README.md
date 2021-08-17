# Neural Network from Scratch

A project to create a multi layer neural network from scratch using just numpy!

## neural_network package contains:

activation_functions.py to define all activation functions and their derivatives

hidden_layer.py to define one hidden layer of the network

mlp.py to define the whole network with a set of hidden layers

## experiments package contains:

Experiments.py to perform experiments with various combinations of network size, activation functions and optimizations

Train vs validation accuracy and loss plots for all experiments

One final experiment on mnist dataset for submission to kaggle

## test package contains:

Comparison test cases between the numpy implementaion and a similar keras model

## exceptions package contains:

User defined exceptions to raise during neural network creation like "shape mismatch between input data and no. of input layer neurons etc.
