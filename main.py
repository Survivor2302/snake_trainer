from snake import *
from vue import *
import genetic 
from Utils import *
import sys

#paramètres de l'évaluation : nombre de parties + taille de la grille
gameParams={"nbGames":10, "height":10, "width":10}

#fonction d'optimisation, renvoie un réseau de neurones entrainé sur le jeu
nn = genetic.optimize(taillePopulation=400, tailleSelection=50, pc=0.8, arch=[nbFeatures, 24, nbActions], gameParams=gameParams, nbIterations=400, nbThreads=10, scoreMax=1.0)
#sauvegarde du réseau pour utilisation en inférence
save_nn(nn, "model.txt")

"""
Pour charger directement, on commente l'optimisation et : 
n = load_nn("model.txt")
"""

#Test visuel, on voit le réseau jouer en temps réel
vue = SnakeVue(gameParams["height"], gameParams["width"], 64)
fps = pygame.time.Clock()
gameSpeed = 20

while True:
    game = Game(gameParams["height"], gameParams["width"])
    while game.enCours:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                pygame.quit()
                sys.exit(0)
        pred = nn.predict(game.getFeatures())
        game.direction = pred
        game.refresh()
        if not game.enCours: break
        vue.displayGame(game)
        fps.tick(gameSpeed)
        if game.steps>100: break

