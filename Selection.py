import random
from copy import deepcopy
from Individual import Individual

class Selection:
    def __init__(self):
        pass

    def tournament_selection(self, population: list[Individual]):
        choices = []
        for i in range(3):
            r = random.randint(0, len(population) - 1)
            choices.append(population[r])
        choices.sort(key=Individual.sortKey, reverse=True)
        return deepcopy(choices[0])

    def roulette_selection(self, population: list[Individual]):

        total_fitness = sum(ind.fitness for ind in population)

        r = random.uniform(0, total_fitness)
        cumulative = 0

        for ind in population:
            cumulative += ind.fitness
            if cumulative >= r:
                return deepcopy(ind)