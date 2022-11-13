# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 21:23:31 2020

@author: Bautista
"""
import numpy as np

class Loss():
    def __call__():
        pass
    def gradient():
        # If called without arguments, it is calculated for the value of the last call
        pass

class MSE_xor(Loss):
    def __call__(self, s, y_true):
        self.grad = 2*(s-y_true)
        self.value = np.mean(self.grad*self.grad/4)
        self.grad/=s.shape[0]
        return self.value
        
    def gradient(self, s = None, y_true = None):
        if s == None or y_true == None:
            return self.grad
        return 2*(s-y_true)/s.shape[0]
    
class MSE(Loss):
    def __call__(self, s, y_true):
        self.grad = 2*(s-y_true)
        self.value = np.mean(np.sum(self.grad*self.grad/4, axis = 1))
        self.grad/=s.shape[0]
        return self.value
        
    def gradient(self, s = None, y_true = None):
        if s == None or y_true == None:
            return self.grad
        return 2*(s-y_true)/s.shape[0]
    
class CCE(Loss):
    def __call__(self, s, y_true):
         self.y_class=np.argmax(y_true, axis = 1)
         self.norm = np.sum(np.exp(s.astype(np.float64)), axis = 1)
         self.s = s
         self.value = -np.mean(np.log(np.exp(s[np.arange(s.shape[0]), self.y_class])/self.norm))
         return self.value
     
    def gradient(self, s = None, y_true=None):
        if s == None or y_true == None:
            self.grad = np.exp(self.s)/self.norm[:, np.newaxis]
            self.grad[np.arange(self.s.shape[0]), self.y_class] -= 1
            self.grad/=self.s.shape[0]
            return self.grad
        self.norm = np.sum(np.exp(s), axis = 1)
        self.grad = np.exp(s)/self.norm[:, np.newaxis]
        self.grad[np.arange(s.shape[0]), y_true] -= 1
        self.grad/=s.shape[0]
        return self.grad










