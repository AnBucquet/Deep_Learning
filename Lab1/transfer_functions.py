# transfer functions


import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# derivative of sigmoid

def dsigmoid(x):
    y = sigmoid(x)
    return y * (1.0 - y)


def tanh(x):
    return np.tanh(x)

# derivative of tanh

def dtanh(x):
    y = tanh(x)
    return 1 - y*y


def identity(x):
    return x

# derivative of identity

def didentity(x):
    return np.ones(x.shape)


def relu(x):
    return (x + np.sign(x)*x)/2

# derivative of relu

def drelu(x):
    return (1 + np.sign(x))/2

def softmax(x):
    K = np.tile(np.reshape(np.sum(np.exp(x), axis=1), [x.shape[0], 1]), [1, x.shape[1]])
    return np.exp(x)/K