from Individual import Individual
from copy import deepcopy
import random

class Crossover:
    def __init__(self, _CROSSOVER_RATE):
        self.CROSSOVER_RATE = _CROSSOVER_RATE

    #for single dna one chromosome
    def single_cut_single_DNA_one_chromosome(self, parent1: Individual, parent2: Individual):
        child = deepcopy(parent1)
        dna1 = parent1.dnaType.DNA
        dna2 = parent2.dnaType.DNA
        cut = random.randint(0, min(len(dna1) - 1, len(dna2) - 1))
        child.dnaType.DNA = dna1[:cut] + dna2[cut:]
        return child

    def multi_cut_single_DNA_one_chromosome(self, parent1: Individual, parent2: Individual):
        child = deepcopy(parent1)
        return child

    def random_bit_choosing_single_DNA_one_chromosome(self, parent1: Individual, parent2: Individual):
        child = deepcopy(parent1)
        return child

    def weighted_bit_choosing_single_DNA_one_chromosome(self, parent1: Individual, parent2: Individual):
        child = deepcopy(parent1)
        return child
    #---------------------------


    def single_cut_diplo_DNA_one_chromosome(self, parent1: Individual, parent2: Individual):
        child = deepcopy(parent1)
        return child

    def single_cut_single_DNA_multi_chromosome(self, parent1: Individual, parent2: Individual):
        child = deepcopy(parent1)
        return child