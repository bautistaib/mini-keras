# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 17:35:17 2020

@author: Bautista
"""
import numpy as np


def  add_ones(x):
    return np.hstack((np.ones((x.shape[0],1)), x))


class Optimizer():
    def __init__(self, lr = 1e-4):
        self.lr = lr
    
    def __call__(self):
        pass
    
class SGD(Optimizer):
    def __init__(self, lr = 1e-4, bsize = 20):
        self.lr = lr
        self.bsize = bsize
    
    def __call__(self, model, x, y, loss):
        nbatch = np.int(x.shape[0]/self.bsize)
        idx = np.arange(x.shape[0])
        np.random.shuffle(idx)
        for btc in range(nbatch):
            xb = x[idx[btc*self.bsize: (btc+1)*self.bsize]]
            yb = y[idx[btc*self.bsize: (btc+1)*self.bsize]]
            loss(model.forward_upto(model.nlayers, xb), yb)
            model.backward(xb, yb, opt = self, loss = loss)
    

    def get_dw(self, layer, local_grad):
        if layer.regularizer == None:
            return -self.lr*(np.dot(add_ones(layer.input).T, local_grad*layer.activation.gradient()))
        return -self.lr*(np.dot(add_ones(layer.input).T, local_grad*layer.activation.gradient()) + layer.regularizer.gradient(layer.get_W()))


            