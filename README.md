# Mini-Keras

Mini-Keras is a minimalistic reimplementation of the popular Keras deep learning library. This project is designed to provide a deeper understanding of the internal workings of Keras by recreating its basic functionality from scratch.

## Features

Mini-Keras includes the following components:

- **Layers**: Basic building blocks of neural networks, including fully connected (Dense) layers and input layers.
- **Activation Functions**: Functions that determine the output of a neuron, including Sigmoid, Tanh, ReLU, Leaky ReLU, and Linear.
- **Loss Functions**: Functions that measure the difference between the predicted and actual values, including Mean Squared Error (MSE) and Categorical Cross-Entropy (CCE).
- **Metrics**: Functions that evaluate the performance of the model, including MSE and Accuracy.
- **Optimizers**: Algorithms that update the weights of the model during training, including Stochastic Gradient Descent (SGD).
- **Regularizers**: Functions that apply penalties on layer parameters during optimization, including L1 and L2 regularization.

## Usage

An example of how to use Mini-Keras to train a neural network model on the CIFAR-10 dataset is provided in the `example.py` file. This includes preprocessing the data, defining the model, and fitting the model to the training data with different learning rates.

## Disclaimer

Please note that Mini-Keras is a simplified version of Keras and does not include all of its features. It is intended for educational purposes only, to provide a deeper understanding of how Keras works under the hood. For serious deep learning projects, we recommend using the full-featured Keras library.

