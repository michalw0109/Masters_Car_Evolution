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


        def connection_based(self, DNA: Single_DNA_one_chromosome):
            """
            Skanuje DNA (lista bitów) i zwraca strukturę sieci (lista połączeń).
            Zwraca listę słowników: [{'source': int, 'target': int, 'weight': float}, ...]
            """

            nn = []

            for idx in range(0, len(DNA.DNA) - 23, 24):
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

        def connection_based_markers(self, DNA: Single_DNA_one_chromosome):
            nn = []

            i = 0
            limit = len(DNA.DNA) - 31

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

        def matrix_connections(self, DNA: Single_DNA_one_chromosome):

            nn = []
            if len(DNA.DNA) < 256:
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


            for idx in range(0, len(weights_bits) - 7, 8):
                weight_bits = weights_bits[idx:idx + 8]
                raw_weight = bits_to_int(weight_bits)
                weight = (raw_weight / 255.0) * 10 - 5

                if idx/8 >= len(nn):
                    return nn

                nn[int(idx/8)]["weight"] = weight



            return nn


        def triangular_matrix_connections(self, DNA: Single_DNA_one_chromosome):

            nn = []
            if len(DNA.DNA) < 120:
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

            for idx in range(0, len(weights_bits) - 7, 8):
                weight_bits = weights_bits[idx:idx + 8]

                raw_weight = bits_to_int(weight_bits)
                weight = (raw_weight / 255.0) * 10 - 5
                if idx / 8 >= len(nn):
                    return nn

                nn[int(idx/8)]["weight"] = weight


            return nn


        def fixed_topology(self, DNA: Single_DNA_one_chromosome):

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

            for idx in range(0, len(DNA.DNA) - 7, 8):
                weight_bits = DNA.DNA[idx:idx + 8]
                raw_weight = bits_to_int(weight_bits)
                weight = (raw_weight / 255.0) * 10 - 5

                if idx/8 >= len(nn):
                    return nn

                nn[int(idx/8)]["weight"] = weight

            return nn


        def grammar_matrix(self, DNA: Single_DNA_one_chromosome):

            # tutaj problem 000 rozwala sie na 4 * 000, potem kolejna interacja i mamy 4x4 macierz symboli czyli 16*000, potem w teori mozna 8x8(64 symbole 000) i czrawty raz na
            # 16x16 symboli 000 -> mocny overkill, można po drugiej iteracji 4x4 macierzy symboli 000 zrobic z nowych reguł tym razem 8 symboli mapuje sie na 16 bitów czyli na 16
            # symboli i powstaje 8x8 macierz z symboli 0000 czyli mozna od razu mapping zrobic na 16x16 bitową o co nam chodzi

            nn = []

            if len(DNA.DNA) < 224:
                return nn

            def get_rule(bits, idx, size):
                return bits[idx * size:(idx + 1) * size]

            grammar_rules = DNA.DNA[:96]  # 8 * 12
            final_rules = DNA.DNA[96:224]  # 8 * 16
            weights_bits = DNA.DNA[224:]

            matrix_symbols = [0, 0, 0]  # start symbol

            # --- grammar expansion (2 iterations) ---
            for _ in range(2):
                new_symbols = []

                for i in range(0, len(matrix_symbols), 3):
                    symbol = matrix_symbols[i:i + 3]
                    idx = bits_to_int(symbol)

                    rule = get_rule(grammar_rules, idx, 12)
                    new_symbols.extend(rule)

                matrix_symbols = new_symbols

            # now we have 16 symbols (4x4), each 3 bits

            matrix_bits = []

            # --- symbol → 16 bit expansion ---
            for i in range(0, len(matrix_symbols), 3):
                symbol = matrix_symbols[i:i + 3]
                idx = bits_to_int(symbol)

                rule = get_rule(final_rules, idx, 16)

                matrix_bits.extend(rule)


            for i in range(16):
                for j in range(16):
                    if matrix_bits[i * 16 + j] == 1:
                        nn.append({
                            'source': i,
                            'target': j,
                            'weight': 0
                        })

            for idx in range(0, len(weights_bits) - 7, 8):
                weight_bits = weights_bits[idx:idx + 8]
                raw_weight = bits_to_int(weight_bits)
                weight = (raw_weight / 255.0) * 10 - 5

                if idx / 8 >= len(nn):
                    return nn

                nn[int(idx / 8)]["weight"] = weight

            return nn




        def cellular_division(self, DNA: Single_DNA_one_chromosome):

            cells = [{'id': 0}]
            connections = []


            for idx in range(0, len(DNA.DNA) - 24, 25):

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


    class decodes_double_DNA_one_chromosome:

        def __init__(self, inputs: list[int], outputs: list[int], marker=None):
            if marker is None:
                marker = [0, 1, 1, 1, 1, 1, 1, 1]

            self.INPUTS = inputs
            self.OUTPUTS = outputs
            self.MARKER = marker

            self.NR_OF_NEURONS  = self.OUTPUTS[len(self.OUTPUTS) - 1] + 1
