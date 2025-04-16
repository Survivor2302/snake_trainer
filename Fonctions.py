import numpy as np

def identite(x):
    return x

def d_identite(x, y):
    return 1

def relu(x):
    return np.where(x>=0, x, 0)

def d_relu(x, y):
    return np.where(x>=0, 1, 0)

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def d_sigmoid(x, y):
    return y*(1.0 - y)

def tanh(x):
    return np.tanh(x)

def d_tanh(x, y):
    return 1.0 - y**2

def elu(x):
    return np.where(x>=0, x, 0.1*(np.exp(x)-1))

def d_elu(x, y):
    return np.where(x>=0, 1, y + 0.1)