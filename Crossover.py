from Individual import Individual
from copy import deepcopy
import random

class Crossover:
    def __init__(self, _CROSSOVER_RATE):
        self.CROSSOVER_RATE = _CROSSOVER_RATE

    #for single dna one chromosome
    def single_cut_single_DNA_one_chromosome(self, parent1: Individual, parent2: Individual):
        child = deepcopy(parent1)
        if random.random() < self.CROSSOVER_RATE:
            dna1 = parent1.dnaType.DNA
            dna2 = parent2.dnaType.DNA
            if min(len(dna1) - 1, len(dna2) - 1) == 0:
                return child
            cut = random.randint(0, min(len(dna1) - 1, len(dna2) - 1))
            child.dnaType.DNA = dna1[:cut] + dna2[cut:]
        return child

    def multi_cut_single_DNA_one_chromosome(self, parent1: Individual, parent2: Individual):
        child = deepcopy(parent1)
        if random.random() < self.CROSSOVER_RATE:
            dna1 = parent1.dnaType.DNA
            dna2 = parent2.dnaType.DNA
            if min(len(dna1) - 1, len(dna2) - 1) == 0:
                return child
            cut1 = random.randint(0, min(len(dna1) - 1, len(dna2) - 1))
            cut2 = random.randint(0, min(len(dna1) - 1, len(dna2) - 1))
            if cut1 > cut2:
                (cut1, cut2) = (cut2, cut1)
            child.dnaType.DNA = dna1[:cut1] + dna2[cut1:cut2] + dna1[cut2:]
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