from Individual import Individual
from copy import deepcopy
import random
from utils import connectionToDNA, generateRandomDna

class Crossover:
    def __init__(self):
        pass

    def no_crossover(self, parent1: Individual, parent2: Individual):
        child = deepcopy(parent1)
        return child

    class crossover_single_DNA_one_chromosome:
        def __init__(self, _CROSSOVER_RATE, _MARKER = None):
            self.CROSSOVER_RATE = _CROSSOVER_RATE
            self.MARKER = _MARKER

        #### SIMPLE CUT BASED ####

        ### RANDOM ###
        def single_cut(self, parent1: Individual, parent2: Individual):
            child = deepcopy(parent1)
            if random.random() < self.CROSSOVER_RATE:
                dna1 = parent1.dnaType.DNA
                dna2 = parent2.dnaType.DNA
                if min(len(dna1) - 1, len(dna2) - 1) == 0:
                    return child
                cut = random.randint(0, min(len(dna1) - 1, len(dna2) - 1))
                child.dnaType.DNA = dna1[:cut] + dna2[cut:]
            return child

        def multi_cut(self, parent1: Individual, parent2: Individual):
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

        def random_bit(self, parent1: Individual, parent2: Individual):
            child = deepcopy(parent1)
            if random.random() < self.CROSSOVER_RATE:
                dna1 = parent1.dnaType.DNA
                dna2 = parent2.dnaType.DNA
                min_len = min(len(dna1), len(dna2))
                new_dna = [dna1[i] if random.random() < 0.5 else dna2[i] for i in range(min_len)]
                new_dna.extend(dna1[min_len:] if len(dna1) > len(dna2) else dna2[min_len:])
                child.dnaType.DNA = new_dna
            return child

        ### WEIGHTED ###

        def single_cut_weighted(self, parent1: Individual, parent2: Individual):
            child = deepcopy(parent1)
            if random.random() < self.CROSSOVER_RATE:
                dna1 = parent1.dnaType.DNA
                dna2 = parent2.dnaType.DNA
                min_len = min(len(dna1), len(dna2))
                if min_len == 0:
                    return child
                f1, f2 = parent1.fitness, parent2.fitness
                total = f1 + f2
                w1 = f1 / total if total > 0 else 0.5
                # cut point biased toward end when parent1 is fitter (more dna1 in child)
                cut = int(random.triangular(0, min_len, w1 * min_len))
                child.dnaType.DNA = dna1[:cut] + dna2[cut:]
            return child

        def multi_cut_weighted(self, parent1: Individual, parent2: Individual):
            child = deepcopy(parent1)
            if random.random() < self.CROSSOVER_RATE:
                dna1 = parent1.dnaType.DNA
                dna2 = parent2.dnaType.DNA
                min_len = min(len(dna1), len(dna2))
                if min_len == 0:
                    return child
                f1, f2 = parent1.fitness, parent2.fitness
                total = f1 + f2
                w2 = f2 / total if total > 0 else 0.5
                # middle segment from parent2 sized proportional to its fitness
                seg = int(random.triangular(0, min_len, w2 * min_len))
                cut1 = random.randint(0, max(0, min_len - seg))
                cut2 = min(cut1 + seg, min_len)
                child.dnaType.DNA = dna1[:cut1] + dna2[cut1:cut2] + dna1[cut2:]
            return child

        def random_bit_weighted(self, parent1: Individual, parent2: Individual):
            child = deepcopy(parent1)
            if random.random() < self.CROSSOVER_RATE:
                dna1 = parent1.dnaType.DNA
                dna2 = parent2.dnaType.DNA
                f1, f2 = parent1.fitness, parent2.fitness
                total = f1 + f2
                w1 = f1 / total if total > 0 else 0.5
                min_len = min(len(dna1), len(dna2))
                new_dna = [dna1[i] if random.random() < w1 else dna2[i] for i in range(min_len)]
                new_dna.extend(dna1[min_len:] if f1 >= f2 else dna2[min_len:])
                child.dnaType.DNA = new_dna
            return child


        #### smart gene-aware crossovers ####

        def _w(self, f1, f2):
            total = f1 + f2
            return f1 / total if total > 0 else 0.5

        def connection_based(self, parent1: Individual, parent2: Individual):
            """Pick each 24-bit connection gene from either parent with fitness-weighted probability."""
            child = deepcopy(parent1)
            if random.random() < self.CROSSOVER_RATE:
                dna1 = parent1.dnaType.DNA
                dna2 = parent2.dnaType.DNA
                f1, f2 = parent1.fitness, parent2.fitness
                w1 = self._w(f1, f2)
                min_len = min(len(dna1), len(dna2))
                new_dna = []
                for idx in range(0, min_len - 23, 24):
                    gene = dna1[idx:idx+24] if random.random() < w1 else dna2[idx:idx+24]
                    new_dna.extend(gene)
                processed = (min_len // 24) * 24
                new_dna.extend(dna1[processed:] if f1 >= f2 else dna2[processed:])
                child.dnaType.DNA = new_dna
            return child

        def connection_based_markers(self, parent1: Individual, parent2: Individual):

            MARKER_LEN = 8
            GENE_LEN = 32  # marker (8) + 24 bity danych

            def is_gene_at(dna, i):
                return i <= len(dna) - GENE_LEN and dna[i:i + MARKER_LEN] == self.MARKER

            child = deepcopy(parent1)

            if random.random() >= self.CROSSOVER_RATE:
                return child

            dna1 = parent1.dnaType.DNA
            dna2 = parent2.dnaType.DNA

            f1, f2 = parent1.fitness, parent2.fitness
            w1 = self._w(f1, f2)  # prawdopodobieństwo wyboru parent1

            i1, i2 = 0, 0
            new_dna = []

            while i1 < len(dna1) and i2 < len(dna2):

                g1 = is_gene_at(dna1, i1)
                g2 = is_gene_at(dna2, i2)

                # --- CASE 1: oba junk ---
                if not g1 and not g2:
                    if random.random() < w1:
                        new_dna.append(dna1[i1])
                    else:
                        new_dna.append(dna2[i2])
                    i1 += 1
                    i2 += 1

                # --- CASE 2: gene vs junk ---
                elif g1 and not g2:
                    if random.random() < w1:
                        new_dna.extend(dna1[i1:i1 + GENE_LEN])
                    # jeśli nie wybrany → ignorujemy gen (śmieci)
                    i1 += GENE_LEN
                    i2 += 1

                elif not g1 and g2:
                    if random.random() < (1 - w1):
                        new_dna.extend(dna2[i2:i2 + GENE_LEN])
                    i1 += 1
                    i2 += GENE_LEN

                # --- CASE 3: gene vs gene ---
                else:
                    if random.random() < w1:
                        new_dna.extend(dna1[i1:i1 + GENE_LEN])
                    else:
                        new_dna.extend(dna2[i2:i2 + GENE_LEN])

                    i1 += GENE_LEN
                    i2 += GENE_LEN

            # opcjonalnie: dopełnienie końcówki
            # (jeśli jeden rodzic dłuższy)
            tail = dna1[i1:] if random.random() < w1 else dna2[i2:]
            new_dna.extend(tail)

            child.dnaType.DNA = new_dna
            return child

        def matrix_connections(self, parent1: Individual, parent2: Individual):
            """
            Choose entire 256-bit connection matrix from one parent (weighted),
            then mix weight tail 8 bits at a time.
            """
            child = deepcopy(parent1)
            if random.random() < self.CROSSOVER_RATE:
                dna1 = parent1.dnaType.DNA
                dna2 = parent2.dnaType.DNA
                if len(dna1) < 256 or len(dna2) < 256:
                    return child
                f1, f2 = parent1.fitness, parent2.fitness
                w1 = self._w(f1, f2)
                new_dna = list(dna1[:256] if random.random() < w1 else dna2[:256])
                min_tail = min(len(dna1), len(dna2))
                for idx in range(256, min_tail - 7, 8):
                    new_dna.extend(dna1[idx:idx+8] if random.random() < w1 else dna2[idx:idx+8])
                processed = 256 + ((max(min_tail - 256, 0)) // 8) * 8
                new_dna.extend(dna1[processed:] if f1 >= f2 else dna2[processed:])
                child.dnaType.DNA = new_dna
            return child

        def triangular_matrix_connections(self, parent1: Individual, parent2: Individual):
            """
            Choose entire 120-bit triangular connection matrix from one parent (weighted),
            then mix weight tail 8 bits at a time.
            """
            child = deepcopy(parent1)
            if random.random() < self.CROSSOVER_RATE:
                dna1 = parent1.dnaType.DNA
                dna2 = parent2.dnaType.DNA
                if len(dna1) < 120 or len(dna2) < 120:
                    return child
                f1, f2 = parent1.fitness, parent2.fitness
                w1 = self._w(f1, f2)
                new_dna = list(dna1[:120] if random.random() < w1 else dna2[:120])
                min_tail = min(len(dna1), len(dna2))
                for idx in range(120, min_tail - 7, 8):
                    new_dna.extend(dna1[idx:idx+8] if random.random() < w1 else dna2[idx:idx+8])
                processed = 120 + ((max(min_tail - 120, 0)) // 8) * 8
                new_dna.extend(dna1[processed:] if f1 >= f2 else dna2[processed:])
                child.dnaType.DNA = new_dna
            return child

        def fixed_topology(self, parent1: Individual, parent2: Individual):

            child = deepcopy(parent1)
            if random.random() < self.CROSSOVER_RATE:
                dna1 = parent1.dnaType.DNA
                dna2 = parent2.dnaType.DNA
                f1, f2 = parent1.fitness, parent2.fitness
                w1 = self._w(f1, f2)
                min_len = min(len(dna1), len(dna2))
                new_dna = []
                for idx in range(0, min_len - 7, 8):
                    instr = dna1[idx:idx + 8] if random.random() < w1 else dna2[idx:idx + 8]
                    new_dna.extend(instr)
                processed = (min_len // 8) * 8
                new_dna.extend(dna1[processed:] if f1 >= f2 else dna2[processed:])
                child.dnaType.DNA = new_dna
            return child

        def grammar_matrix(self, parent1: Individual, parent2: Individual):
            """
            Mix 8 grammar rules (12 bits each) individually,
            mix 8 final rules (16 bits each) individually,
            then mix weight tail 8 bits at a time.
            """
            child = deepcopy(parent1)
            if random.random() < self.CROSSOVER_RATE:
                dna1 = parent1.dnaType.DNA
                dna2 = parent2.dnaType.DNA
                if len(dna1) < 224 or len(dna2) < 224:
                    return child
                f1, f2 = parent1.fitness, parent2.fitness
                w1 = self._w(f1, f2)
                new_dna = []
                for i in range(8):                          # grammar rules: 8 × 12 bits
                    s = i * 12
                    new_dna.extend(dna1[s:s+12] if random.random() < w1 else dna2[s:s+12])
                for i in range(8):                          # final rules:   8 × 16 bits
                    s = 96 + i * 16
                    new_dna.extend(dna1[s:s+16] if random.random() < w1 else dna2[s:s+16])
                min_tail = min(len(dna1), len(dna2))
                for idx in range(224, min_tail - 7, 8):     # weight tail:   8 bits each
                    new_dna.extend(dna1[idx:idx+8] if random.random() < w1 else dna2[idx:idx+8])
                processed = 224 + ((max(min_tail - 224, 0)) // 8) * 8
                new_dna.extend(dna1[processed:] if f1 >= f2 else dna2[processed:])
                child.dnaType.DNA = new_dna
            return child

        def cellular_division(self, parent1: Individual, parent2: Individual):
            """Pick each 25-bit instruction from either parent with fitness-weighted probability."""
            child = deepcopy(parent1)
            if random.random() < self.CROSSOVER_RATE:
                dna1 = parent1.dnaType.DNA
                dna2 = parent2.dnaType.DNA
                f1, f2 = parent1.fitness, parent2.fitness
                w1 = self._w(f1, f2)
                min_len = min(len(dna1), len(dna2))
                new_dna = []
                for idx in range(0, min_len - 24, 25):
                    instr = dna1[idx:idx+25] if random.random() < w1 else dna2[idx:idx+25]
                    new_dna.extend(instr)
                processed = (min_len // 25) * 25
                new_dna.extend(dna1[processed:] if f1 >= f2 else dna2[processed:])
                child.dnaType.DNA = new_dna
            return child

        # def make_phenotype_crossover(self, decoder):################################ to do poprawy, nie ma prawa dzialac bo potrzeba wtedy enkodera sieci do dna, a to nie fajne
        #     """
        #     Returns a crossover function that works at the phenotype level.
        #     Both parents are decoded to connection lists; connections are merged by (source, target):
        #       - both have it   → pick one weighted by fitness
        #       - only parent1   → include with probability w1
        #       - only parent2   → include with probability w2
        #     The child DNA is re-encoded as a pure 24-bit-per-connection sequence
        #     (compatible with connection_based_no_markers decoder).
        #     """
        #     def crossover(parent1: Individual, parent2: Individual):
        #         child = deepcopy(parent1)
        #         if random.random() < self.CROSSOVER_RATE:
        #             conn1 = decoder(parent1.dnaType)
        #             conn2 = decoder(parent2.dnaType)
        #             f1, f2 = parent1.fitness, parent2.fitness
        #             w1 = self._w(f1, f2)
        #
        #             dict1 = {(c["source"], c["target"]): c for c in conn1}
        #             dict2 = {(c["source"], c["target"]): c for c in conn2}
        #
        #             selected = []
        #             for key in set(dict1) | set(dict2):
        #                 in1, in2 = key in dict1, key in dict2
        #                 if in1 and in2:
        #                     selected.append(dict1[key] if random.random() < w1 else dict2[key])
        #                 elif in1:
        #                     if random.random() < w1:
        #                         selected.append(dict1[key])
        #                 else:
        #                     if random.random() < (1 - w1):
        #                         selected.append(dict2[key])
        #
        #             new_dna = []
        #             for conn in selected:
        #                 new_dna.extend(connectionToDNA(conn))
        #             child.dnaType.DNA = new_dna
        #         return child
        #
        #     return crossover

    #---------------------------
    class crossover_double_DNA_one_chromosome:
        def __init__(self, _CROSSOVER_RATE, _MARKER = None):
            self.CROSSOVER_RATE = _CROSSOVER_RATE
            self.MARKER = _MARKER


        def single_cut(self, parent1: Individual, parent2: Individual):
            child = deepcopy(parent1)
            return child

