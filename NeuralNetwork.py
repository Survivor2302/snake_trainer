import numpy as np
import random
from Dense import *
from Fonctions import *

class NeuralNetwork:
    def __init__(self, inputShape):
        self.inputShape = inputShape
        self.layers = []

    def addLayer(self, size, activationFunctionStr):
        activationFunction = relu
        d_activationFunction = d_relu
        match activationFunctionStr:
            case "logistic":
                activationFunction = sigmoid
                d_activationFunction = d_sigmoid
            case "relu":
                activationFunction = relu
                d_activationFunction = d_relu
            case "elu":
                activationFunction = elu
                d_activationFunction = d_elu
            case "identity":
                activationFunction = identite
                d_activationFunction = d_identite
            case "tanh":
                activationFunction = tanh
                d_activationFunction = d_tanh
            case _:
                activationFunction = sigmoid
                d_activationFunction = d_sigmoid
        inputShape = self.inputShape if len(self.layers)==0 else self.layers[-1].outputShape
        self.layers.append(Dense(inputShape, (size,), activationFunction, d_activationFunction))
    
    def compute(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer.compute(outputs)
        return outputs

    def predict(self, inputs):
        return np.argmax(self.compute(inputs))

