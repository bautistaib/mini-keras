# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 21:30:21 2020

@author: Bautista
"""
import numpy as np
import Models, Metrics, Layers, Losses, Activations, Optimizers, Regularizers
from keras.datasets import mnist, cifar10
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# preprocessing
white = np.zeros(x_train[:10].shape)
y_train = np.reshape(y_train, y_train.size)
y_test = np.reshape(y_test, y_test.size)
x_train = np.reshape(x_train, (x_train.shape[0], np.prod(x_train.shape[1:]))).astype(np.float64)
x_test = np.reshape(x_test, (x_test.shape[0], np.prod(x_test.shape[1:]))).astype(np.float64)

# change the y target to one hot encoding
yy_train = np.zeros((x_train.shape[0], 10)).astype(np.uint8)
yy_train[np.arange(x_train.shape[0]), y_train] = 1
yy_test = np.zeros((x_test.shape[0], 10)).astype(np.uint8)
yy_test[np.arange(x_test.shape[0]), y_test] = 1
mean = np.mean(x_train, axis=0)[None,:]
x_test = (x_test - mean)/255
x_train = (x_train - mean)/255


# Defining the model
reg = Regularizers.L2(7e-4)
model = Models.Network()
model.add(Layers.InputLayer(x_train.shape[1]))
model.add(Layers.Dense(units = 100, w_std = 1e-3, activation = Activations.LReLU(), input_dim = x_train.shape[1], regularizer = reg))
model.add(Layers.Dense(units = 100, w_std = 1e-3, activation = Activations.LReLU(), regularizer=reg))
model.add(Layers.Dense(units = 10, w_std = 1e-3, activation = Activations.Linear(), regularizer = reg))

#we fit the model with different learting rates
model.fit(x_train, yy_train, x_test, yy_test, epochs = 10, loss = Losses.CCE(), metric = Metrics.accuracy, opt = Optimizers.SGD(lr=2e-1, bsize = 20))
model.fit(x_train, yy_train, x_test, yy_test, epochs = 30, loss = Losses.CCE(), metric = Metrics.accuracy, opt = Optimizers.SGD(lr=1e-2, bsize = 20))
model.fit(x_train, yy_train, x_test, yy_test, epochs = 70, loss = Losses.CCE(), metric = Metrics.accuracy, opt = Optimizers.SGD(lr=1e-3, bsize = 20))
