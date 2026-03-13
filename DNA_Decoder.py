from utils import bits_to_int
from DNA import *

class Decoder:
    def __init__(self):
        pass

    class decodes_single_DNA_one_chromosome:

        def __init__(self, inputs: list[int], outputs: list[int], marker=None):
            if marker is None:
                marker = [0, 1, 1, 1, 1, 1, 1, 1]

            self.INPUTS = inputs
            self.OUTPUTS = outputs
            self.MARKER = marker

            self.NR_OF_NEURONS  = self.OUTPUTS[len(self.OUTPUTS) - 1] + 1


        def connection_based_no_markers_no_grey(self, DNA: Single_DNA_one_chromosome):
            """
            Skanuje DNA (lista bitów) i zwraca strukturę sieci (lista połączeń).
            Zwraca listę słowników: [{'source': int, 'target': int, 'weight': float}, ...]
            """

            nn = []

            for idx in range(0, len(DNA.DNA) - 24, 24):
                # Dekodowanie
                source_bits = DNA.DNA[idx: idx + 8]
                target_bits = DNA.DNA[idx + 8: idx + 16]
                weight_bits = DNA.DNA[idx + 16: idx + 24]

                source_id = bits_to_int(source_bits) % self.NR_OF_NEURONS
                target_id = bits_to_int(target_bits) % self.NR_OF_NEURONS

                # Mapowanie wagi z 0-255 na -5.0 do 5.0
                raw_weight = bits_to_int(weight_bits)
                weight = (raw_weight / 255.0) * 10.0 - 5.0

                nn.append({
                    'source': source_id,
                    'target': target_id,
                    'weight': weight
                })

            return nn

        def connection_based_with_markers_no_grey(self, DNA: Single_DNA_one_chromosome):
            nn = []

            i = 0
            limit = len(DNA.DNA) - 32

            while i < limit:
                # Sprawdź czy tutaj jest marker
                if DNA.DNA[i: i + 8] == self.MARKER:
                    # Znaleziono gen! Przeskakujemy marker
                    idx = i + 8

                    source_bits = DNA.DNA[idx: idx + 8]
                    target_bits = DNA.DNA[idx + 8: idx + 16]
                    weight_bits = DNA.DNA[idx + 16: idx + 24]

                    source_id = bits_to_int(source_bits) % self.NR_OF_NEURONS
                    target_id = bits_to_int(target_bits) % self.NR_OF_NEURONS

                    # Mapowanie wagi z 0-255 na -5.0 do 5.0
                    raw_weight = bits_to_int(weight_bits)
                    weight = (raw_weight / 255.0) * 10.0 - 5.0

                    nn.append({
                        'source': source_id,
                        'target': target_id,
                        'weight': weight
                    })

                    # Przeskocz przeczytany gen
                    i += 32
                else:
                    # To nie marker, idziemy bit dalej (to jest junk dna)
                    i += 1

            return nn

        def matrix_16x16_connections_no_grey(self, DNA: Single_DNA_one_chromosome):

            nn = []
            if len(DNA.DNA) > 256:
                return nn

            matrix_bits = DNA.DNA[:256]
            weights_bits = DNA.DNA[256:]


            for i in range(16):
                for j in range(16):
                    if matrix_bits[i * 16 + j] == 1:
                        nn.append({
                            'source': i,
                            'target': j,
                            'weight': 0
                        })


            for idx in range(0, len(weights_bits) - 8, 8):
                weight_bits = weights_bits[idx:idx + 8]
                raw_weight = bits_to_int(weight_bits)
                weight = (raw_weight / 255.0) * 10 - 5

                if idx/8 >= len(nn):
                    return nn

                nn[int(idx/8)]["weight"] = weight



            return nn


        def triangular_matrix_connections_no_grey(self, DNA: Single_DNA_one_chromosome):

            nn = []
            if len(DNA.DNA) > 120:
                return nn

            matrix_bits = DNA.DNA[:120]
            weights_bits = DNA.DNA[120:]

            bit_idx = 0

            for i in range(16):
                for j in range(i + 1, 16):

                    bit = matrix_bits[bit_idx]
                    bit_idx += 1

                    if bit == 1:

                        nn.append({
                            'source': i,
                            'target': j,
                            'weight': 0
                        })

            for idx in range(0, len(weights_bits) - 8, 8):
                weight_bits = weights_bits[idx:idx + 8]

                raw_weight = bits_to_int(weight_bits)
                weight = (raw_weight / 255.0) * 10 - 5
                if idx / 8 >= len(nn):
                    return nn

                nn[int(idx/8)]["weight"] = weight


            return nn


        def grammar_matrix_no_grey(self, DNA: Single_DNA_one_chromosome):

            nn = []
            if len(DNA.DNA) > 96:
                return nn

            def get_rule(idx, rules):
                return rules[idx*12:(idx+1)*12]

            rules_bits = DNA.DNA[:96]

            weights_bits = DNA.DNA[96:]

            matrix_bits = [0,0,0]

            for _ in range(4):
                new_matrix_bits = []
                for i in range(0, len(matrix_bits), 3):
                    symbol = matrix_bits[i:i+3]
                    rule = get_rule(bits_to_int(symbol), rules_bits)
                    new_matrix_bits.append(rule)
                matrix_bits = new_matrix_bits


            for i in range(16):
                for j in range(16):
                    if matrix_bits[i * 16 + j] == 1:
                        nn.append({
                            'source': i,
                            'target': j,
                            'weight': 0
                        })

            for idx in range(0, len(weights_bits) - 8, 8):
                weight_bits = weights_bits[idx:idx + 8]
                raw_weight = bits_to_int(weight_bits)
                weight = (raw_weight / 255.0) * 10 - 5

                if idx / 8 >= len(nn):
                    return nn

                nn[int(idx / 8)]["weight"] = weight

            return nn

        def fixed_topology_weights_no_grey(self, DNA: Single_DNA_one_chromosome):

            nn = []

            # input->hidden
            for i in range(len(self.INPUTS)):
                for j in range(len(self.INPUTS), self.NR_OF_NEURONS - len(self.OUTPUTS)):
                    nn.append({
                        'source': i,
                        'target': j,
                        'weight': 0
                    })

            # hidden->output
            for i in range(len(self.INPUTS), self.NR_OF_NEURONS - len(self.OUTPUTS)):
                for j in range(self.NR_OF_NEURONS - len(self.OUTPUTS), self.NR_OF_NEURONS):
                    nn.append({
                        'source': i,
                        'target': j,
                        'weight': 0
                    })

            for idx in range(0, len(DNA.DNA) - 8, 8):
                weight_bits = DNA.DNA[idx:idx + 8]
                raw_weight = bits_to_int(weight_bits)
                weight = (raw_weight / 255.0) * 10 - 5

                if idx/8 >= len(nn):
                    return nn

                nn[int(idx/8)]["weight"] = weight

            return nn

        def cellular_division_no_grey(self, DNA: Single_DNA_one_chromosome):

            cells = [{'id': 0}]
            connections = []


            for idx in range(0, len(DNA.DNA) - 25, 25):

                opcode = DNA.DNA[idx]

                if opcode == 0:
                    cell = bits_to_int(DNA.DNA[idx + 1:idx + 9]) % self.NR_OF_NEURONS
                    cells.append({'id': cell})

                else:

                    cell = bits_to_int(DNA.DNA[idx + 1:idx + 9]) % len(cells)

                    target = bits_to_int(DNA.DNA[idx + 9:idx + 17]) % len(cells)

                    raw = bits_to_int(DNA.DNA[idx + 17:idx + 25])
                    weight = (raw / 255) * 10 - 5

                    connections.append({
                        'source': cells[cell]['id'],
                        'target': cells[target]['id'],
                        'weight': weight
                    })

            return connections
