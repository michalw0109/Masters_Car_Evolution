import random
from utils import *
from DNA import *

class Initializer:
    def __init__(self):
        pass

    class init_single_DNA_one_chromosome:

        def __init__(self, inputs: list[int], outputs: list[int], marker=None):
            if marker is None:
                marker = [0, 1, 1, 1, 1, 1, 1, 1]

            self.INPUTS = inputs
            self.OUTPUTS = outputs
            self.MARKER = marker

            self.NR_OF_NEURONS  = self.OUTPUTS[len(self.OUTPUTS) - 1] + 1



        def random_bits(self):
            DNA: Single_DNA_one_chromosome = Single_DNA_one_chromosome()
            DNA.DNA = generateRandomDna(random.randint(600, 800))
            return DNA

        def connection_based(self):

            nrOfGoodConnections = random.randint(5, 20)
            junk_size = 24
            DNA: Single_DNA_one_chromosome = Single_DNA_one_chromosome()

            for _ in range(nrOfGoodConnections):
                while True:
                    input = bits_to_int(generateRandomDna(8))
                    if input % self.NR_OF_NEURONS in self.INPUTS:
                        break
                while True:
                    output = bits_to_int(generateRandomDna(8))
                    if output % self.NR_OF_NEURONS in self.OUTPUTS:
                        break

                DNA.DNA.extend(connectionToDNA({'source': input, 'target': output, 'weight': random.random() * 10 - 5}))
                DNA.DNA.extend(generateRandomDna(junk_size))
            return DNA

        def connection_based_markers(self):

            nrOfGoodConnections = random.randint(5, 20)
            DNA: Single_DNA_one_chromosome = Single_DNA_one_chromosome()
            DNA.DNA.extend(generateRandomDna(random.randint(1, 36)))

            for _ in range(nrOfGoodConnections):
                while True:
                    input = bits_to_int(generateRandomDna(8))
                    if input % self.NR_OF_NEURONS in self.INPUTS:
                        break
                while True:
                    output = bits_to_int(generateRandomDna(8))
                    if output % self.NR_OF_NEURONS in self.OUTPUTS:
                        break

                DNA.DNA.extend(self.MARKER)
                DNA.DNA.extend(connectionToDNA({'source': input, 'target': output, 'weight': random.random() * 10 - 5}))
                DNA.DNA.extend(generateRandomDna(random.randint(1, 36)))
            return DNA

        def matrix_connections(self):

            DNA: Single_DNA_one_chromosome = Single_DNA_one_chromosome()

            DNA.DNA.extend([random.choice([0, 0, 0, 0, 0, 0, 0, 1]) for _ in range(256)])
            DNA.DNA.extend(generateRandomDna(random.randint(200, 400)))


            return DNA


        def triangular_matrix_connections(self):

            DNA: Single_DNA_one_chromosome = Single_DNA_one_chromosome()

            DNA.DNA.extend([random.choice([0, 0, 0, 1]) for _ in range(120)])
            DNA.DNA.extend(generateRandomDna(random.randint(200, 400)))

            return DNA

        def fixed_topology(self):

            DNA: Single_DNA_one_chromosome = Single_DNA_one_chromosome()


            # input->hidden
            for i in range(len(self.INPUTS)):
                for j in range(len(self.INPUTS), self.NR_OF_NEURONS - len(self.OUTPUTS)):
                    DNA.DNA.extend(generateRandomDna(8))

            # hidden->output
            for i in range(len(self.INPUTS), self.NR_OF_NEURONS - len(self.OUTPUTS)):
                for j in range(self.NR_OF_NEURONS - len(self.OUTPUTS), self.NR_OF_NEURONS):
                    DNA.DNA.extend(generateRandomDna(8))

            return DNA

        def grammar_matrix(self):

            DNA: Single_DNA_one_chromosome = Single_DNA_one_chromosome()

            DNA.DNA.extend(generateRandomDna(96))
            DNA.DNA.extend(generateRandomDna(128))
            DNA.DNA.extend(generateRandomDna(random.randint(400, 800)))


            return DNA



        def cellular_division(self):

            nrOfCells = random.randint(16, 24)
            nrOfConnections = random.randint(5, 20)
            DNA: Single_DNA_one_chromosome = Single_DNA_one_chromosome()

            for _ in range(nrOfCells):
                DNA.DNA.extend([0])
                DNA.DNA.extend(generateRandomDna(24))
                DNA.DNA.extend([1])
                DNA.DNA.extend(generateRandomDna(24))


            for _ in range(nrOfConnections):
                DNA.DNA.extend([1])
                DNA.DNA.extend(generateRandomDna(24))

            return DNA


    class init_double_DNA_one_chromosome:

        def __init__(self, inputs: list[int], outputs: list[int], marker=None):
            if marker is None:
                marker = [0, 1, 1, 1, 1, 1, 1, 1]

            self.INPUTS = inputs
            self.OUTPUTS = outputs
            self.MARKER = marker

            self.NR_OF_NEURONS  = self.OUTPUTS[len(self.OUTPUTS) - 1] + 1

        # ── internal helper: delegate to init_single and extract its .DNA ──
        def _make_single(self):
            return Initializer.init_single_DNA_one_chromosome(self.INPUTS, self.OUTPUTS, self.MARKER)

        def _new(self) -> Double_DNA_one_chromosome:
            return Double_DNA_one_chromosome()

        # Both strands are initialised independently using the matching single
        # initialiser so that each strand starts as a valid encoding on its own.

        def random_bits(self) -> Double_DNA_one_chromosome:
            s = self._make_single()
            dna = self._new()
            dna.DNAa = s.random_bits().DNA
            dna.DNAb = s.random_bits().DNA
            return dna

        def connection_based(self) -> Double_DNA_one_chromosome:
            s = self._make_single()
            dna = self._new()
            dna.DNAa = s.connection_based().DNA
            dna.DNAb = s.connection_based().DNA
            return dna

        def connection_based_markers(self) -> Double_DNA_one_chromosome:
            s = self._make_single()
            dna = self._new()
            dna.DNAa = s.connection_based_markers().DNA
            dna.DNAb = s.connection_based_markers().DNA
            return dna

        def matrix_connections(self) -> Double_DNA_one_chromosome:
            s = self._make_single()
            dna = self._new()
            dna.DNAa = s.matrix_connections().DNA
            dna.DNAb = s.matrix_connections().DNA
            return dna

        def triangular_matrix_connections(self) -> Double_DNA_one_chromosome:
            s = self._make_single()
            dna = self._new()
            dna.DNAa = s.triangular_matrix_connections().DNA
            dna.DNAb = s.triangular_matrix_connections().DNA
            return dna

        def fixed_topology(self) -> Double_DNA_one_chromosome:
            s = self._make_single()
            dna = self._new()
            dna.DNAa = s.fixed_topology().DNA
            dna.DNAb = s.fixed_topology().DNA
            return dna

        def grammar_matrix(self) -> Double_DNA_one_chromosome:
            s = self._make_single()
            dna = self._new()
            dna.DNAa = s.grammar_matrix().DNA
            dna.DNAb = s.grammar_matrix().DNA
            return dna

        def cellular_division(self) -> Double_DNA_one_chromosome:
            s = self._make_single()
            dna = self._new()
            dna.DNAa = s.cellular_division().DNA
            dna.DNAb = s.cellular_division().DNA
            return dna