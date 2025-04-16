import numpy as np
from NeuralNetwork import *
import Fonctions

def to_2D(X):
    X_ = []
    dim = int(np.sqrt(len(X[0])))
    for i in range(len(X)):
        X_.append(X[i].reshape((dim,dim)))
    return np.array(X_)

def to_1D(X):
    X_ = []
    for i in range(len(X)):
        X_.append(X[i].flatten())
    return np.array(X_)


def to_catogorical(y):
    nbClasses = len(np.unique(y))
    y_bool = np.zeros((len(y), nbClasses))
    for idx, val in enumerate(y): y_bool[idx][val] = 1.0
    return y_bool

def load_nn(filename):
    with open(filename, "r") as fichier:
        lines = fichier.readlines()
        fichier.close()

        layerSizes = [int(n) for n in lines[0][:-1].split(" ")]
        activations = lines[1].rstrip().split(" ")
        nn = NeuralNetwork((layerSizes[0],))
        for i in range(1, len(layerSizes)): nn.addLayer(layerSizes[i], activations[i])

        line = 2
        for i in range(len(nn.layers)):
            nn.layers[i].bias = [float(n) for n in lines[line][:-1].split(" ")]
            line+=1
            for j in range(nn.layers[i].inputShape[0]):
                weights = [float(n) for n in lines[line][:-1].split(" ")]
                for k in range(nn.layers[i].outputShape[0]):
                    nn.layers[i].weights[k][j] = weights[k]
                line+=1
        return nn
    
def save_nn(nn: NeuralNetwork, filename):
    with open(filename, "w") as fichier:
        fichier.write(" ".join([str(nn.inputShape[0])]+[str(layer.outputShape[0]) for layer in nn.layers])+"\n")
        activations = ["none"]
        for layer in nn.layers:
            match layer.activationFunction:
                case Fonctions.identite : activations.append("identity")
                case Fonctions.sigmoid : activations.append("logistic")
                case Fonctions.relu : activations.append("relu")
                case Fonctions.elu : activations.append("elu")
                case Fonctions.tanh : activations.append("tanh")
        fichier.write(" ".join(activations)+"\n")
        for i in range(len(nn.layers)):
            fichier.write(" ".join([str(bias) for bias in nn.layers[i].bias])+"\n")
            for j in range(nn.layers[i].inputShape[0]):
                fichier.write(" ".join([str(weight) for weight in nn.layers[i].weights[j]])+"\n")

    
