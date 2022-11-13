# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 21:17:58 2020

@author: Bautista
"""
import numpy as np

def MSE(s, y_true):
    return np.mean(np.sum((s-y_true)*(s-y_true), axis = 1))

def accuracy(s, y_true):
    y_class=np.argmax(y_true, axis = 1)
    pred = np.argmax(s, axis = 1)
    return np.mean(pred == y_class)

def acc_xor(s, y_true):
    sc = np.copy(s)
    sc[np.where(sc>0.9)] = 1 
    sc[np.where(sc<-0.9)] = -1
    return np.mean(sc == y_true)