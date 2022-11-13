# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 00:30:18 2020

@author: Bautista
"""
import numpy as np
import Activations


def  add_ones(x):
    # adds a column of ones to the right of x (for bias)
    return np.hstack((np.ones((x.shape[0],1)), x))


class Layer():
    def __init__(self):
        self.w = None
        self.regularizer = None
        pass
    def __call__(self):
        pass
    
class WLayer(Layer):
    def __init__(self, units = 2, w_std = 1e-3, activation = Activations.ReLU(), input_dim = None, regularizer  = None):
        self.activation = activation
        self.units = units
        self.input_dim = input_dim
        self.w_std = w_std
        if input_dim is not None:
            self.w = np.random.normal(0, np.sqrt(2/(self.input_dim+1+self.units)), (input_dim+1, self.units))
        self.regularizer = regularizer
        
    def set_input_dim(self, input_dim):
        self.input_dim = input_dim
        self.w = np.random.normal(0, np.sqrt(2/(self.input_dim+1+self.units)), (input_dim+1, self.units))
    
    def get_y(self, x):
        return add_ones(x).dot(self.w)
    
    def update_w(self, dw):
        self.w += dw
    
    def get_W(self):
        return self.w
    
    def output_dim(self):
        return self.units
    
class Dense(WLayer):
    # def __init__(self, units = 2, w_std = 1e-4, activation = Activations.Tanh(), input_dim = None, regularizador  = None):
    #     super().__init__(self, units = 2, w_std = 1e-4, activation = Activations.Tanh(), input_dim = None, regularizador  = None)
    def __call__(self, x):
        self.input = x
        return self.activation(self.get_y(x))
    
    def get_type(self):
        return "Dense"
    def get_input(self):
        return self.input
    
    def get_gradient(self, local_grad, x = None):
        if x is None:
            grad = np.copy(local_grad)
            grad *= self.activation.gradient()
            grad = np.dot(grad, self.w.T)
            return grad[:,1:]
        grad = np.copy(local_grad)
        grad *= self.activation.gradient(x)
        grad = np.dot(grad, self.w.T)
        return grad[:, 1:]
    
    def inverse(self, s):
        inv = self.activation.inverse(s)
        inv = inv.dot(np.linalg.pinv(self.w))
        inv = inv[:,1:]
        return inv
            
    


class ConcatInput(Layer):
    def __init__(self, concat, input_dim = None):
        self.concat_dim = concat.output_dim()
        self.concat = concat
        self.input_dim = input_dim
        self.w =None
        self.regularizer = None
    
    def set_input_dim(self, input_dim):
        self.input_dim = input_dim
    
    def output_dim(self):
        return self.concat_dim + self.input_dim
    
    def __call__(self, s):
        return np.hstack((s, self.concat(self.concat.get_input())))
    
    def get_gradient(self, local_grad, x = None):
        if x is None:
            return local_grad[:, :self.input_dim]
        return local_grad[:, :x.shape[1]]
    
    def get_type(self):
        return "ConcatInput"
        
        

class InputLayer(Layer):
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.w = None
        self.regularizer = None

    
    def output_dim(self):
        return self.input_dim
    
    def __call__(self, x):
        self.x = x
        return self.x
    
    def get_input(self):
        return self.x
    
    def get_type(self):
        return "Input"
    
    def get_gradient(self, local_grad, x = None):
        return local_grad
    
    def inverse(self, s):
        return s
    

         

