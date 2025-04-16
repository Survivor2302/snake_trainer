import random
import itertools
import numpy
from NeuralNetwork import *

nbFeatures = 8
nbActions = 4

class Game:
    def __init__(self, hauteur, largeur):
        self.grille = [[0]*hauteur  for _ in range(largeur)]
        self.hauteur, self.largeur = hauteur, largeur
        self.serpent = [[largeur//2-i-1, hauteur//2] for i in range(4)]
        for (x,y) in self.serpent: self.grille[x][y] = 1
        self.direction = 3
        self.accessibles = [[x,y] for (x,y) in list(itertools.product(range(largeur), range(hauteur))) if [x,y] not in self.serpent]
        self.fruit = [0,0]
        self.setFruit()
        self.enCours = True
        self.steps = 0
        self.score = 4
    
    def setFruit(self):
        if (len(self.accessibles)==0): return
        self.fruit = self.accessibles[random.randint(0, len(self.accessibles)-1)][:]
        self.grille[self.fruit[0]][self.fruit[1]] = 2

    def refresh(self):
        nextStep = self.serpent[0][:]
        match self.direction:
            case 0: nextStep[1]-=1
            case 1: nextStep[1]+=1
            case 2: nextStep[0]-=1
            case 3: nextStep[0]+=1

        if nextStep not in self.accessibles:
            self.enCours = False
            return
        self.accessibles.remove(nextStep)
        if self.grille[nextStep[0]][nextStep[1]]==2:
            self.setFruit()
            self.steps = 0
            self.score+=1
        else:
            self.steps+=1
            self.grille[self.serpent[-1][0]][self.serpent[-1][1]] = 0
            self.accessibles.append(self.serpent[-1][:])
            self.serpent = self.serpent[:-1]
            if self.steps>self.hauteur*self.largeur:
                self.enCours = False
                return

        self.grille[nextStep[0]][nextStep[1]] = 1
        self.serpent = [nextStep]+self.serpent

    def getFeatures(self):
        features = numpy.zeros(8)
        
        # Position de la tête du serpent
        tete = self.serpent[0]
        
        # 1-4: Obstacles dans les 4 directions (haut, bas, gauche, droite)
        # Vérifier si un obstacle est présent au-dessus
        if tete[1] == 0 or [tete[0], tete[1]-1] in self.serpent:
            features[0] = 1
        
        # Vérifier si un obstacle est présent en-dessous
        if tete[1] == self.hauteur-1 or [tete[0], tete[1]+1] in self.serpent:
            features[1] = 1
        
        # Vérifier si un obstacle est présent à gauche
        if tete[0] == 0 or [tete[0]-1, tete[1]] in self.serpent:
            features[2] = 1
        
        # Vérifier si un obstacle est présent à droite
        if tete[0] == self.largeur-1 or [tete[0]+1, tete[1]] in self.serpent:
            features[3] = 1
        
        # 5: Position relative du fruit en Y (au-dessus: 1, en-dessous: -1, même ligne: 0)
        if self.fruit[1] < tete[1]:
            features[4] = 1
        elif self.fruit[1] > tete[1]:
            features[4] = -1
        else:
            features[4] = 0
        
        # 6: Position relative du fruit en X (à droite: 1, à gauche: -1, même colonne: 0)
        if self.fruit[0] > tete[0]:
            features[5] = 1
        elif self.fruit[0] < tete[0]:
            features[5] = -1
        else:
            features[5] = 0
        
        # 7: Direction actuelle du serpent
        features[6] = self.direction
        
        # 8: Distance jusqu'au bord dans la direction actuelle
        if self.direction == 0:  # Haut
            features[7] = tete[1]
        elif self.direction == 1:  # Bas
            features[7] = self.hauteur - 1 - tete[1]
        elif self.direction == 2:  # Gauche
            features[7] = tete[0]
        elif self.direction == 3:  # Droite
            features[7] = self.largeur - 1 - tete[0]
        
        return features
    
    def print(self):
        print("".join(["="]*(self.largeur+2)))
        for ligne in range(self.hauteur):
            chaine = ["="]
            for colonne in range(self.largeur):
                if self.grille[colonne][ligne]==1: chaine.append("#")
                elif self.grille[colonne][ligne]==2: chaine.append("F")
                else: chaine.append(" ")
            chaine.append("=")
            print("".join(chaine))
        print("".join(["="]*(self.largeur+2))+"\n")

