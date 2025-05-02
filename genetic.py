import numpy
import time
import multiprocessing as mp
from NeuralNetwork import *
from snake import *

# Pool de processus global pour le parallélisme
_process_pool = None


def init_pool():
    """
    Initialise le pool de processus pour le parallélisme.
    Crée 10 processus pour évaluer les parties en parallèle.
    """
    global _process_pool
    if _process_pool is None:
        _process_pool = mp.Pool(processes=10)


def close_pool():
    """
    Ferme proprement le pool de processus.
    Attend que tous les processus soient terminés avant de continuer.
    """
    global _process_pool
    if _process_pool is not None:
        _process_pool.close()
        _process_pool.join()
        _process_pool = None


def eval_single_game(params):
    """
    Évalue un réseau de neurones sur une partie de Snake.

    Args:
        params (dict): Dictionnaire contenant les paramètres de la partie et le réseau de neurones

    Returns:
        float: Score de la partie (score * 1000 + nombre de pas)
    """
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
    """
    Évalue un lot de solutions en parallèle.

    Args:
        solutions (list): Liste des individus à évaluer
        gameParams (dict): Paramètres du jeu

    Returns:
        list: Liste des scores normalisés des solutions
    """
    global _process_pool

    nbGames = gameParams["nbGames"]
    hauteur = gameParams["height"]
    largeur = gameParams["width"]

    # Préparation des paramètres pour l'évaluation parallèle
    all_params = []
    for sol in solutions:
        for _ in range(nbGames):
            params = {
                "hauteur": hauteur,
                "largeur": largeur,
                "nn": sol.nn
            }
            all_params.append(params)

    # Exécution parallèle des parties
    if _process_pool is None:
        init_pool()

    all_scores = _process_pool.map(eval_single_game, all_params)

    # Calcul des scores moyens normalisés
    for i, sol in enumerate(solutions):
        scores = all_scores[i * nbGames:(i + 1) * nbGames]
        total_score = sum(scores)
        sol.score = total_score / (nbGames * hauteur * largeur * 1000)

    return [sol.score for sol in solutions]


def eval(sol, gameParams):
    """
    Évalue un seul individu.

    Args:
        sol (Individu): L'individu à évaluer
        gameParams (dict): Paramètres du jeu

    Returns:
        float: Score de l'individu
    """
    return eval_batch([sol], gameParams)[0]


def optimize(taillePopulation, tailleSelection, pc, arch, gameParams, nbIterations, nbThreads, scoreMax):
    """
    Algorithme génétique principal pour optimiser les réseaux de neurones.

    Args:
        taillePopulation (int): Taille de la population
        tailleSelection (int): Nombre d'individus à sélectionner
        pc (float): Probabilité de croisement
        arch (tuple): Architecture du réseau de neurones
        gameParams (dict): Paramètres du jeu
        nbIterations (int): Nombre d'itérations maximum
        nbThreads (int): Nombre de threads (non utilisé)
        scoreMax (float): Score maximum à atteindre

    Returns:
        NeuralNetwork: Meilleur réseau de neurones trouvé
    """
    try:
        # Initialisation
        start_time_total = time.time()
        population = initialization(taillePopulation, arch, gameParams)
        mr = 2.0  # Taux de mutation

        # Boucle d'évolution
        for iteration in range(nbIterations):
            start_time_iter = time.time()

            # Condition d'arrêt si score maximum atteint
            if population[0].score >= scoreMax:
                total_time = time.time() - start_time_total
                print(
                    f"Score maximum atteint: {population[0].score:.4f} (Temps total: {total_time:.2f}s)")
                break

            # Sélection des meilleurs individus
            selected = population[:tailleSelection]

            # Génération de nouveaux individus
            new_individuals = []
            children_to_evaluate = []

            while len(new_individuals) < taillePopulation - tailleSelection:
                # Sélection aléatoire des parents
                parent1 = selected[numpy.random.randint(0, tailleSelection)]
                parent2 = selected[numpy.random.randint(0, tailleSelection)]

                # Croisement et mutation
                child1, child2 = crossover(parent1, parent2, pc)
                child1 = mutate(child1, mr)
                child2 = mutate(child2, mr)

                # Ajout des enfants pour évaluation
                children_to_evaluate.extend([child1, child2])
                new_individuals.append(child1)
                if len(new_individuals) < taillePopulation - tailleSelection:
                    new_individuals.append(child2)

            # Évaluation des nouveaux individus
            eval_batch(children_to_evaluate, gameParams)

            # Mise à jour de la population
            population = selected + new_individuals
            population.sort(reverse=True, key=lambda sol: sol.score)

            # Affichage des statistiques
            iter_time = time.time() - start_time_iter
            total_time = time.time() - start_time_total
            print(
                f"Itération {iteration+1}/{nbIterations}, Meilleur score: {population[0].score:.4f} (Temps itération: {iter_time:.2f}s, Temps total: {total_time:.2f}s)")

        # Retour du meilleur réseau
        total_time = time.time() - start_time_total
        print(
            f"Meilleur score final: {population[0].score:.4f} (Temps total: {total_time:.2f}s)")
        return population[0].nn

    finally:
        # Nettoyage
        close_pool()


class Individu:
    """
    Représente un individu dans l'algorithme génétique.

    Attributs:
        nn (NeuralNetwork): Le réseau de neurones de l'individu
        score (float): Score de l'individu (à maximiser)
    """

    def __init__(self, nn):
        self.nn = nn
        self.score = 0

    def __lt__(self, other):
        return self.score < other.score


def initialization(taillePopulation, arch, gameParams):
    """
    Initialise la population initiale.

    Args:
        taillePopulation (int): Taille finale de la population
        arch (tuple): Architecture du réseau de neurones
        gameParams (dict): Paramètres du jeu

    Returns:
        list: Population initiale triée par score
    """
    population = []
    # Génération d'une population plus grande pour sélectionner les meilleurs
    for i in range(taillePopulation * 10):
        nn = NeuralNetwork((arch[0],))
        for j in range(1, len(arch)):
            nn.addLayer(arch[j], "elu")
        population.append(Individu(nn))

    # Évaluation et sélection des meilleurs
    for sol in population:
        eval(sol, gameParams)
    population.sort(reverse=True, key=lambda sol: sol.score)
    return population[:taillePopulation]


def crossover(parent1, parent2, pc):
    """
    Effectue le croisement entre deux parents pour créer deux enfants.

    Args:
        parent1 (Individu): Premier parent
        parent2 (Individu): Deuxième parent
        pc (float): Probabilité de croisement

    Returns:
        tuple: Les deux enfants créés
    """
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


def mutate(individu, mr):
    """
    Applique des mutations aléatoires à un individu.

    Args:
        individu (Individu): L'individu à muter
        mr (float): Taux de mutation

    Returns:
        Individu: L'individu muté
    """
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
