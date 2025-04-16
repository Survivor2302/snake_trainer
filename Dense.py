import numpy as np
import Fonctions


class Dense:
    def __init__(self, inputShape, outputShape, activationFunction, activationDerivate):
        self.inputShape = inputShape
        self.outputShape = outputShape
        self.activationFunction = activationFunction
        self.activationDerivate =  activationDerivate

        self.aggregations = np.zeros(outputShape)        
        self.outputs = np.zeros(outputShape)

        self.bias =  np.zeros(outputShape)
        # Choix de l'initialisation en fonction de la fonction d'activation
        if activationFunction == Fonctions.relu:
            # Initialisation He pour ReLU
            std = np.sqrt(2.0 / inputShape[0])
        elif activationFunction == Fonctions.tanh or activationFunction == Fonctions.sigmoid:
            # Initialisation Xavier pour tanh
            std = np.sqrt(1.0 / inputShape[0])
        else:
            std = 1.0 / np.sqrt(inputShape[0])
            
        self.weights = np.random.randn(inputShape[0], outputShape[0]) * std

        self.inputErrors = np.zeros(inputShape)
        
    def compute(self, inputs):
        self.inputs = inputs
        self.aggregations = np.dot(inputs, self.weights)+self.bias
        self.outputs = self.activationFunction(self.aggregations)
        return self.outputs
    