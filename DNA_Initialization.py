import random
from utils import *
from DNA import *

class Initializer:
    def __init__(self):
        pass

    def connection_based_single_DNA_one_chromosome_no_markers_no_grey(self):

        nrOfGoodConnections = random.randint(5, 20)
        junk_size = 24
        DNA: Single_DNA_one_chromosome = Single_DNA_one_chromosome()
        INPUTS = [0, 1, 2, 3, 4, 5]
        OUTPUTS = [12, 13, 14, 15]
        for _ in range(nrOfGoodConnections):
            while True:
                input = bits_to_int(generateRandomDna(8)) % 16
                if input in INPUTS:
                    break
            while True:
                output = bits_to_int(generateRandomDna(8)) % 16
                if output in OUTPUTS:
                    break
            DNA.DNA.extend(connectionToDNA({'source': input, 'target': output, 'weight': random.random() * 10 - 5}))
            DNA.DNA.extend(generateRandomDna(junk_size))
        return DNA

    def random_bits_single_DNA_one_chromosome(self):
        DNA: Single_DNA_one_chromosome = Single_DNA_one_chromosome()
        return DNA
