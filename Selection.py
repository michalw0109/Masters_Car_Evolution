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

    def roulette_selection(self):
        pass


