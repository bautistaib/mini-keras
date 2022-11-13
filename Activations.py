# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 23:22:19 2020

@author: Bautista
"""
import numpy as np


class Activation():
    def __call__():
        pass
    def gradient():
        # If called without arguments, it is calculated for the value of the last call
        pass
    
class Sigmoid(Activation):
    def __call__(self, x):
        self.value = 1/(1+np.exp(-x))
        return self.value
    
    def gradient(self, x = None):
        if x is None:
            self.grad = self.value*(1-self.value)
            return self.grad
        self.grad = np.exp(x)/((1+np.exp(x))**2)
        return self.grad
    
    def inverse(self, s):
        return np.log(s/(1-s))


class Tanh(Activation):
    def __call__(self, x):
        self.value = np.tanh(x)
        return self.value
    
    def gradient(self, x = None):
        if x is None:
            self.grad = 1-self.value*self.value
            return self.grad
        self.grad = 1-np.tanh(x)*np.tanh(x)
        return self.grad
    
    def inverse(self, s):
        return np.arctanh(s)

class ReLU(Activation):
    def __call__(self, x):
        self.value = np.maximum(0,x)
        return self.value
    
    def gradient(self, x = None):
        if x is None:
            self.grad = np.heaviside(self.value, 0)
            return self.grad
        self.grad = np.heaviside(np.maximum(x, 0), 0)
        return self.grad
    
class LReLU(Activation):
    def __call__(self, x):
        self.value = np.maximum(0.1*x,x)
        return self.value
    
    def gradient(self, x = None):
        if x is None:
            self.grad = 0.1*(self.value<=0)+1*(self.value>0)
            return self.grad
        self.grad = 0.1*(x<=0)+1*(x>0)
        return self.grad
    
    def inverse(self, s):
        return np.minimum(10*s, s)


class Linear(Activation):
    def __call__(self, x):
        self.value = x
        return self.value
    
    def gradient(self, x=None):
        if x is None:
            self.grad = np.ones(self.value.shape)
            return self.grad
        self.grad = np.ones(x.shape)
        return self.grad
    
    def inverse(self, s):
        return s



