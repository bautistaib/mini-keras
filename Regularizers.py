# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 21:38:58 2020

@author: Bautista
"""
import numpy as np

class Regularizer:
    def __call__(self):
        pass
    def gradient(self):
        pass

class L2(Regularizer):
    def __init__(self, lbda):
        self.lbda = lbda
    def __call__(self, w):
        return self.lbda*0.5*np.sum(w*w)
    def gradient(self, w):
        return self.lbda*w

class L1(Regularizer):
    def __init__(self, lbda):
        self.lbda = lbda
    def __call__(self, w):
        return self.lbda*np.sum(np.abs(w))
    def gradient(self, w):
        return self.lbda*np.sign(w)
    