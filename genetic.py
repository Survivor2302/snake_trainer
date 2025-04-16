import numpy
import time
import multiprocessing as mp
from NeuralNetwork import *
from snake import *

# Variable globale pour le pool de processus
_process_pool = None


def init_pool():
    global _process_pool
    if _process_pool is None:
        # Fixé à 10 processus pour les 10 parties
        _process_pool = mp.Pool(processes=10)


def close_pool():
    global _process_pool
    if _process_pool is not None:
        _process_pool.close()
        _process_pool.join()
        _process_pool = None


def eval_single_game(params):
    game = Game(params["hauteur"], params["largeur"])
    moves_in_right_direction = 0
    last_distance = abs(
        game.serpent[0][0] - game.fruit[0]) + abs(game.serpent[0][1] - game.fruit[1])
    max_steps_without_fruit = params["hauteur"] * params["largeur"]
    nn = params["nn"]

    while game.enCours:
        features = game.getFeatures()
        game.direction = nn.predict(features)
        game.refresh()

        if game.enCours:
            new_distance = abs(
                game.serpent[0][0] - game.fruit[0]) + abs(game.serpent[0][1] - game.fruit[1])
            if new_distance < last_distance:
                moves_in_right_direction += 1
            last_distance = new_distance

        if game.steps > max_steps_without_fruit:
            break

    return 1000 * game.score + game.steps


def eval_batch(solutions, gameParams):
    global _process_pool

    nbGames = gameParams["nbGames"]
    hauteur = gameParams["height"]
    largeur = gameParams["width"]

    # Préparation des paramètres pour tous les individus et leurs parties
    all_params = []
    for sol in solutions:
        for _ in range(nbGames):
            params = {
                "hauteur": hauteur,
                "largeur": largeur,
                "nn": sol.nn
            }
            all_params.append(params)

    # Exécution parallèle de toutes les parties
    if _process_pool is None:
        init_pool()

    all_scores = _process_pool.map(eval_single_game, all_params)

    # Distribution des scores aux solutions
    for i, sol in enumerate(solutions):
        scores = all_scores[i * nbGames:(i + 1) * nbGames]
        total_score = sum(scores)
        sol.score = total_score / (nbGames * hauteur * largeur * 1000)

    return [sol.score for sol in solutions]


def eval(sol, gameParams):
    return eval_batch([sol], gameParams)[0]


def optimize(taillePopulation, tailleSelection, pc, arch, gameParams, nbIterations, nbThreads, scoreMax):
    try:
        # Initialisation de la population
        start_time_total = time.time()
        population = initialization(taillePopulation, arch, gameParams)

        # Taux de mutation
        mr = 2.0

        # Boucle principale de l'algorithme génétique
        for iteration in range(nbIterations):
            start_time_iter = time.time()

            # Si le meilleur score dépasse le seuil, on arrête
            if population[0].score >= scoreMax:
                total_time = time.time() - start_time_total
                print(
                    f"Score maximum atteint: {population[0].score:.4f} (Temps total: {total_time:.2f}s)")
                break

            # Sélection des meilleurs individus
            selected = population[:tailleSelection]

            # Génération de nouveaux individus par croisement et mutation
            new_individuals = []
            children_to_evaluate = []

            while len(new_individuals) < taillePopulation - tailleSelection:
                # Sélection aléatoire de deux parents parmi les meilleurs
                parent1 = selected[numpy.random.randint(0, tailleSelection)]
                parent2 = selected[numpy.random.randint(0, tailleSelection)]

                # Croisement
                child1, child2 = crossover(parent1, parent2, pc)

                # Mutation
                child1 = mutate(child1, mr)
                child2 = mutate(child2, mr)

                # Ajout des enfants pour évaluation groupée
                children_to_evaluate.extend([child1, child2])
                new_individuals.append(child1)
                if len(new_individuals) < taillePopulation - tailleSelection:
                    new_individuals.append(child2)

            # Évaluation groupée des enfants
            eval_batch(children_to_evaluate, gameParams)

            # Fusion de la population sélectionnée et des nouveaux individus
            population = selected + new_individuals

            # Tri de la population par score décroissant
            population.sort(reverse=True, key=lambda sol: sol.score)

            # Calcul et affichage du temps
            iter_time = time.time() - start_time_iter
            total_time = time.time() - start_time_total
            print(
                f"Itération {iteration+1}/{nbIterations}, Meilleur score: {population[0].score:.4f} (Temps itération: {iter_time:.2f}s, Temps total: {total_time:.2f}s)")

        total_time = time.time() - start_time_total
        print(
            f"Meilleur score final: {population[0].score:.4f} (Temps total: {total_time:.2f}s)")
        return population[0].nn

    finally:
        # S'assurer que le pool est fermé à la fin
        close_pool()


'''
Représente une solution avec
_un réseau de neurones
_un score (à maximiser)

vous pouvez ajouter des attributs ou méthodes si besoin
'''


class Individu:
    def __init__(self, nn):
        self.nn = nn
        self.score = 0

    def __lt__(self, other):
        return self.score < other.score


'''
La méthode d'initialisation de la population est donnée :
_on génère N individus contenant chacun un réseau de neurones (de même format)
_on évalue et on trie des individus
'''


def initialization(taillePopulation, arch, gameParams):
    population = []
    for i in range(taillePopulation):
        nn = NeuralNetwork((arch[0],))
        for j in range(1, len(arch)):
            nn.addLayer(arch[j], "elu")
        population.append(Individu(nn))

    for sol in population:
        eval(sol, gameParams)
    population.sort(reverse=True, key=lambda sol: sol.score)

    return population

# Fonction de croisement entre deux parents pour générer deux enfants


def crossover(parent1, parent2, pc):
    # Tirage aléatoire pour déterminer si on fait un croisement
    if numpy.random.random() > pc:
        # Pas de croisement, on clone les parents
        child1 = Individu(NeuralNetwork((parent1.nn.inputShape[0],)))
        child2 = Individu(NeuralNetwork((parent2.nn.inputShape[0],)))

        # Copie des couches du premier parent pour le premier enfant
        for i, layer in enumerate(parent1.nn.layers):
            size = layer.outputShape[0]
            activation = "elu"  # Par défaut
            # Déterminer la fonction d'activation utilisée
            if layer.activationFunction == sigmoid:
                activation = "logistic"
            elif layer.activationFunction == relu:
                activation = "relu"
            elif layer.activationFunction == elu:
                activation = "elu"
            elif layer.activationFunction == tanh:
                activation = "tanh"
            elif layer.activationFunction == identite:
                activation = "identity"

            child1.nn.addLayer(size, activation)
            # Copie des poids et biais
            child1.nn.layers[i].weights = numpy.copy(layer.weights)
            child1.nn.layers[i].bias = numpy.copy(layer.bias)

        # Copie des couches du deuxième parent pour le deuxième enfant
        for i, layer in enumerate(parent2.nn.layers):
            size = layer.outputShape[0]
            activation = "elu"  # Par défaut
            # Déterminer la fonction d'activation utilisée
            if layer.activationFunction == sigmoid:
                activation = "logistic"
            elif layer.activationFunction == relu:
                activation = "relu"
            elif layer.activationFunction == elu:
                activation = "elu"
            elif layer.activationFunction == tanh:
                activation = "tanh"
            elif layer.activationFunction == identite:
                activation = "identity"

            child2.nn.addLayer(size, activation)
            # Copie des poids et biais
            child2.nn.layers[i].weights = numpy.copy(layer.weights)
            child2.nn.layers[i].bias = numpy.copy(layer.bias)
    else:
        # Croisement
        child1 = Individu(NeuralNetwork((parent1.nn.inputShape[0],)))
        child2 = Individu(NeuralNetwork((parent2.nn.inputShape[0],)))

        # Pour chaque couche, on fait un croisement
        for i in range(len(parent1.nn.layers)):
            layer1 = parent1.nn.layers[i]
            layer2 = parent2.nn.layers[i]

            size = layer1.outputShape[0]
            activation = "elu"  # Par défaut
            # Déterminer la fonction d'activation utilisée
            if layer1.activationFunction == sigmoid:
                activation = "logistic"
            elif layer1.activationFunction == relu:
                activation = "relu"
            elif layer1.activationFunction == elu:
                activation = "elu"
            elif layer1.activationFunction == tanh:
                activation = "tanh"
            elif layer1.activationFunction == identite:
                activation = "identity"

            # Ajout des couches aux enfants
            child1.nn.addLayer(size, activation)
            child2.nn.addLayer(size, activation)

            # Tirage aléatoire d'un alpha entre 0 et 1
            alpha = numpy.random.random()

            # Croisement des poids
            child1.nn.layers[i].weights = alpha * \
                layer1.weights + (1 - alpha) * layer2.weights
            child2.nn.layers[i].weights = (
                1 - alpha) * layer1.weights + alpha * layer2.weights

            # Croisement des biais
            child1.nn.layers[i].bias = alpha * \
                layer1.bias + (1 - alpha) * layer2.bias
            child2.nn.layers[i].bias = (
                1 - alpha) * layer1.bias + alpha * layer2.bias

    return child1, child2

# Fonction de mutation d'un individu


def mutate(individu, mr):
    for i, layer in enumerate(individu.nn.layers):
        # Calcul des probabilités de mutation
        pm_biais = mr / layer.outputShape[0]
        pm_poids = mr / individu.nn.layers[i].inputShape[0]

        # Mutation des biais
        for j in range(layer.outputShape[0]):
            if numpy.random.random() < pm_biais:
                # Ajout d'une petite perturbation au biais
                layer.bias[j] += numpy.random.randn() * 0.1

        # Mutation des poids
        for j in range(layer.inputShape[0]):
            for k in range(layer.outputShape[0]):
                if numpy.random.random() < pm_poids:
                    # Ajout d'une petite perturbation au poids
                    layer.weights[j][k] += numpy.random.randn() * 0.1

    return individu
