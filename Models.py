# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 00:08:39 2020

@author: Bautista
"""
import numpy as np
import Metrics, Layers, Losses, Activations, Optimizers

class Network():
    def __init__(self):
        self.layers = []
        self.layers_type =[]
        self.output_dim = []
        self.nlayers = 0
        self.metric_tr_h = []
        self.loss_tr_h = []
        self.metric_tst_h = []
        self.loss_tst_h = []
        
    def add(self, layer):
        if not layer.input_dim:
            layer.set_input_dim(self.output_dim[-1])
        self.output_dim.append(layer.output_dim())
        self.layers.append(layer)
        self.layers_type.append(layer.get_type)
        self.nlayers +=1
        
    def get_layer(self, n):
        return self.layers[n]
    
    def forward_upto(self, j, x):
        assert self.nlayers >= j
        self.outputs = []
        for n in range(j):
            if n == 0:
                self.outputs.append(self.layers[n](x))
            else:
                self.outputs.append(self.layers[n](self.outputs[-1]))
        return self.outputs[-1]
    
    def predict(self, x):
        return np.argmax(self.forward_upto(self.nlayers, x), axis=1)
    
    def modify_inputs(self, x_mod, y_target, mean, epochs = 100, loss = Losses.MSE(), lr = 1e-4, renormalize = True, metric = None, lbda = 0.5):
        shape = x_mod.shape
        x_mod = np.reshape(x_mod.astype(np.float64), (x_mod.shape[0], np.prod(x_mod.shape[1:])))-mean
        x_mod/=255
        for e in range(epochs):
            if not e%100:
                print("epoca {}".format(e))
                print("Loss = {}".format(loss(self.forward_upto(self.nlayers, x_mod), y_target)))
                if metric is not None:
                    print("Accuracy = {}".format(metric(self.forward_upto(self.nlayers, x_mod), y_target)))
            else:
                loss(self.forward_upto(self.nlayers, x_mod), y_target)
            self.local_grads = []
            self.local_grads.append(loss.gradient())
            for layer in self.layers[::-1]:
                self.local_grads.append(layer.get_gradient(self.local_grads[-1]))
            x_mod-=lr*(self.local_grads[-1] + lbda*np.sign(x_mod)*(-1*(x_mod+1)*(x_mod<-1)+1*(x_mod-1)*(x_mod>1)))
        x_mod*=255
        x_mod += mean
        x_mod = np.reshape(x_mod, shape)
        if renormalize:
            for n in range(x_mod.shape[0]):
                x_mod[n] = ((x_mod[n]-x_mod[n].min())*255/(x_mod[n].max()-x_mod[n].min())).astype(np.int16)
        return x_mod.astype(np.int16)
    
    def fit(self, x_train, y_train, x_test, y_test, epochs = 100, loss = Losses.MSE(), metric = Metrics.accuracy, opt = Optimizers.SGD(lr = 1e-4, bsize = 20), verbose = True):
        for e in range(epochs):
            if verbose:
                print("epoch {}".format(e))
            opt(self, x_train, y_train, loss)
            self.metric_tr_h.append(metric(self.forward_upto(self.nlayers, x_train), y_train))
            self.loss_tr_h.append(loss(self.outputs[-1], y_train)+np.sum([layer.regularizer(layer.get_W()) for layer in self.layers if layer.regularizer is not None]))
            self.metric_tst_h.append(metric(self.forward_upto(self.nlayers, x_test), y_test))
            self.loss_tst_h.append(loss(self.outputs[-1], y_test)+np.sum([layer.regularizer(layer.get_W()) for layer in self.layers if layer.regularizer is not None]))
            if verbose:
                print("test accuracy = {}".format(self.metric_tst_h[-1]))
                print("train accuracy = {}".format(self.metric_tr_h[-1]))
                print("loss = {}".format(self.loss_tr_h[-1]))
        
    def backward(self, x, y, opt = Optimizers.SGD(lr = 1e-4, bsize = 20), loss = Losses.MSE()):
        self.local_grads = []
        self.local_grads.append(loss.gradient())
        for layer in self.layers[::-1]:
            self.local_grads.append(layer.get_gradient(self.local_grads[-1]))
            if layer.w is not None:
                dw = opt.get_dw(layer, self.local_grads[-2])
                layer.update_w(dw)
    
    def inverse(self, s):
        for layer in self.layers[::-1]:
            s = layer.inverse(s)
        return s