# ------------------ IMPORTS ------------------
from EvolutionEngine import EvolutionEngine
import random
import numpy as np
from Individual import Individual
from Computation import Computation
from Crossover import Crossover
from Mutation import Mutation, MutationCombiner
from Selection import Selection
from DNA import *
from DNA_Initialization import Initializer
from DNA_Decoder import Decoder
import sys





def main() -> None:
    r = int(sys.argv[1])
    np.random.seed(r)
    random.seed(r)

    MAX_GENERATIONS = 1000
    POPULATION_SIZE = 300
    ELITE_FRACTION = 0.05
    CROSSOVER_RATE = 0.8
    MUTATION_RATE = 0.01

    INPUTS = [0, 1, 2, 3, 4, 5]
    OUTPUTS = [12, 13, 14, 15]
    MARKER = [0, 1, 1, 1, 1, 1, 1, 1]

    DNA_INITIALIZATION = Initializer().init_single_DNA_one_chromosome(INPUTS, OUTPUTS, MARKER).cellular_division_no_grey
    DNA_DECODER = Decoder().decodes_single_DNA_one_chromosome(INPUTS, OUTPUTS, MARKER).cellular_division_no_grey

    COMPUTATION = Computation(INPUTS, OUTPUTS, MARKER).connection_based_sort_feed_forward

    SELECTION = Selection().tournament_selection

    CROSSOVER = Crossover(CROSSOVER_RATE).multi_cut_single_DNA_one_chromosome

    MUTATION1 = Mutation(MUTATION_RATE).random_bit_flip_single_DNA_one_chromosome
    MUTATION2 = Mutation(MUTATION_RATE * 10).random_insert_single_DNA_one_chromosome
    MUTATION3 = Mutation(MUTATION_RATE * 10).random_delete_single_DNA_one_chromosome
    MUTATION = MutationCombiner([MUTATION1, MUTATION2, MUTATION3]).mutate

    window = EvolutionEngine(MAX_GENERATIONS, POPULATION_SIZE, ELITE_FRACTION, DNA_INITIALIZATION, DNA_DECODER, COMPUTATION, SELECTION, CROSSOVER, MUTATION)
    window.run()

if __name__ == "__main__":
    main()


