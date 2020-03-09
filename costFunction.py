# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 01:53:29 2020

@author: Walter Aburime
"""
import numpy as np
def costFunction(X,Y, theta):
    m = len(Y)
    J = 0
    
    h_theta = X @ theta
    J = ((1/(2*m)) * np.sum(np.square(h_theta - Y)))
    
    return J