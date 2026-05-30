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

                DNA.DNA.extend(self.MARKER)
                DNA.DNA.extend(connectionToDNA({'source': input, 'target': output, 'weight': random.random() * 10 - 5}))
                DNA.DNA.extend(self.MARKER)
                DNA.DNA.extend(generateRandomDna(junk_size))
            return DNA

        def matrix_connections(self):

            nrOfGoodConnections = random.randint(5, 20)
            DNA: Single_DNA_one_chromosome = Single_DNA_one_chromosome()

            matrix_bits = [0] * 256
            for _ in range(nrOfGoodConnections):
                src = random.choice(self.INPUTS)
                tgt = random.choice(self.OUTPUTS)
                matrix_bits[src * 16 + tgt] = 1

                src = random.randint(0, self.NR_OF_NEURONS - 1)
                tgt = random.randint(0, self.NR_OF_NEURONS - 1)
                matrix_bits[src * 16 + tgt] = 1

            DNA.DNA.extend(matrix_bits)
            DNA.DNA.extend(generateRandomDna(nrOfGoodConnections * 2 * 8))
            return DNA


        def triangular_matrix_connections(self):

            nrOfGoodConnections = random.randint(5, 20)
            DNA: Single_DNA_one_chromosome = Single_DNA_one_chromosome()

            matrix_bits = [0] * 120
            for _ in range(nrOfGoodConnections):
                src = random.choice(self.INPUTS)
                tgt = random.choice(self.OUTPUTS)
                bit_idx = 15 * src - src * (src - 1) // 2 + (tgt - src - 1)
                matrix_bits[bit_idx] = 1

                src = random.randint(0, self.NR_OF_NEURONS - 1)
                tgt = random.randint(0, self.NR_OF_NEURONS - 1)
                bit_idx = 15 * src - src * (src - 1) // 2 + (tgt - src - 1)
                matrix_bits[bit_idx] = 1

            DNA.DNA.extend(matrix_bits)
            DNA.DNA.extend(generateRandomDna(nrOfGoodConnections * 2 * 8))
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

            nrOfConnections = random.randint(5, 20)
            DNA: Single_DNA_one_chromosome = Single_DNA_one_chromosome()

            # Decoder starts with cells = [{'id': 0}] (index 0).
            # Emit one cell gene per input neuron, then per output neuron.
            # After all these genes: cells[1..len(INPUTS)] = inputs,
            #                        cells[len(INPUTS)+1..] = outputs
            for neuron_id in self.INPUTS + self.OUTPUTS:
                DNA.DNA.extend([0])
                DNA.DNA.extend(int_to_bits(neuron_id, 8))
                DNA.DNA.extend(generateRandomDna(16))  # remaining bits ignored

            for _ in range(nrOfConnections):
                src_neuron = random.choice(self.INPUTS)
                tgt_neuron = random.choice(self.OUTPUTS)

                # Cell index in the array built above
                src_idx = self.INPUTS.index(src_neuron) + 1
                tgt_idx = len(self.INPUTS) + self.OUTPUTS.index(tgt_neuron) + 1

                DNA.DNA.extend([1])
                DNA.DNA.extend(int_to_bits(src_idx, 8))
                DNA.DNA.extend(int_to_bits(tgt_idx, 8))
                DNA.DNA.extend(generateRandomDna(8))

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
            dna = self._new()
            r = random.randint(600, 800)
            dna.DNAa = generateRandomDna(r)
            dna.DNAb = generateRandomDna(r)
            return dna

        # def connection_based(self) -> Double_DNA_one_chromosome:
        #     s = self._make_single()
        #     dna = self._new()
        #     dna.DNAa = s.connection_based().DNA
        #     dna.DNAb = s.connection_based().DNA
        #     return dna
        #
        # def connection_based_markers(self) -> Double_DNA_one_chromosome:
        #     s = self._make_single()
        #     dna = self._new()
        #     dna.DNAa = s.connection_based_markers().DNA
        #     dna.DNAb = s.connection_based_markers().DNA
        #     return dna
        #
        # def matrix_connections(self) -> Double_DNA_one_chromosome:
        #     s = self._make_single()
        #     dna = self._new()
        #     dna.DNAa = s.matrix_connections().DNA
        #     dna.DNAb = s.matrix_connections().DNA
        #     return dna
        #
        # def triangular_matrix_connections(self) -> Double_DNA_one_chromosome:
        #     s = self._make_single()
        #     dna = self._new()
        #     dna.DNAa = s.triangular_matrix_connections().DNA
        #     dna.DNAb = s.triangular_matrix_connections().DNA
        #     return dna
        #
        # def fixed_topology(self) -> Double_DNA_one_chromosome:
        #     s = self._make_single()
        #     dna = self._new()
        #     dna.DNAa = s.fixed_topology().DNA
        #     dna.DNAb = s.fixed_topology().DNA
        #     return dna
        #
        # def grammar_matrix(self) -> Double_DNA_one_chromosome:
        #     s = self._make_single()
        #     dna = self._new()
        #     dna.DNAa = s.grammar_matrix().DNA
        #     dna.DNAb = s.grammar_matrix().DNA
        #     return dna
        #
        # def cellular_division(self) -> Double_DNA_one_chromosome:
        #     s = self._make_single()
        #     dna = self._new()
        #     dna.DNAa = s.cellular_division().DNA
        #     dna.DNAb = s.cellular_division().DNA
        #     return dna