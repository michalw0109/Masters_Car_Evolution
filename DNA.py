import random
from utils import *

class DNA:
    def __init__(self):
        self.DNA = None
        #self.nn = None


class Single_DNA_one_chromosome(DNA):

    def __init__(self):
        super().__init__()
        self.DNA: list[int] = []


class Double_DNA_one_chromosome(DNA):
    def __init__(self):
        super().__init__()

        self.DNAa: list[int] = []
        self.DNAb: list[int] = []

    def combine(self, method='random', marker=None) -> list[int]:

        if marker is None:
            marker = [0, 1, 1, 1, 1, 1, 1, 1]

        a, b = self.DNAa, self.DNAb
        length = len(a)



        if method == 'random':
            return list(a) if random.random() < 0.5 else list(b)

        elif method == 'crossover':
            result = [a[i] if random.random() < 0.5 else b[i] for i in range(length)]
            return result

        elif method == 'connection_based':
            result = []
            for idx in range(0, length - 23, 24):
                result.extend(a[idx:idx+24] if bits_to_int(a[idx:idx+8]) > bits_to_int(b[idx:idx+8]) else b[idx:idx+24])
            return result

        elif method == 'connection_based_markers':
            ML = len(marker)
            GENE = ML + 24

            def _is_gene(dna, i):
                return i <= len(dna) - GENE and dna[i:i+ML] == marker

            i1, i2 = 0, 0
            result = []
            while i1 < len(a) and i2 < len(b):
                g1 = _is_gene(a, i1)
                g2 = _is_gene(b, i2)
                if not g1 and not g2:
                    result.append(a[i1] if random.random() < 0.5 else b[i2])
                    i1 += 1; i2 += 1
                elif g1 and not g2:
                    if random.random() < 0.5:
                        result.extend(a[i1:i1+GENE])
                    i1 += GENE; i2 += GENE
                elif not g1 and g2:
                    if random.random() < 0.5:
                        result.extend(b[i2:i2+GENE])
                    i1 += GENE; i2 += GENE
                else:
                    result.extend(a[i1:i1+GENE] if random.random() < 0.5 else b[i2:i2+GENE])
                    i1 += GENE; i2 += GENE
            result.extend(a[i1:] if random.random() < 0.5 else b[i2:])
            return result

        elif method == 'matrix_connections':
            if length < 256:
                return list(a)
            result = list(a[:256] if random.random() < 0.5 else b[:256])
            for idx in range(256, length - 7, 8):
                result.extend(a[idx:idx+8] if random.random() < 0.5 else b[idx:idx+8])
            return result

        elif method == 'triangular_matrix_connections':
            if length < 120:
                return list(a)
            result = list(a[:120] if random.random() < 0.5 else b[:120])
            for idx in range(120, length - 7, 8):
                result.extend(a[idx:idx+8] if random.random() < 0.5 else b[idx:idx+8])
            return result

        elif method == 'fixed_topology':
            result = []
            for idx in range(0, length - 7, 8):
                result.extend(a[idx:idx+8] if random.random() < 0.5 else b[idx:idx+8])
            return result

        elif method == 'grammar_matrix':
            if length < 224:
                return list(a)
            result = []
            for i in range(8):
                s = i * 12
                result.extend(a[s:s+12] if random.random() < 0.5 else b[s:s+12])
            for i in range(8):
                s = 96 + i * 16
                result.extend(a[s:s+16] if random.random() < 0.5 else b[s:s+16])
            for idx in range(224, length - 7, 8):
                result.extend(a[idx:idx+8] if random.random() < 0.5 else b[idx:idx+8])
            return result

        elif method == 'cellular_division':
            result = []
            for idx in range(0, length - 24, 25):
                result.extend(a[idx:idx+25] if random.random() < 0.5 else b[idx:idx+25])
            return result

        return list(a)



class Single_DNA_multi_chromosome(DNA):
    def __init__(self):
        super().__init__()

        self.DNA1: list[int] = []
        self.DNA2: list[int] = []
        self.DNA3: list[int] = []




class Double_DNA_multi_chromosome(DNA):
    def __init__(self):
        super().__init__()

        self.DNA1a: list[int] = []
        self.DNA1b: list[int] = []

        self.DNA2a: list[int] = []
        self.DNA2b: list[int] = []

        self.DNA3a: list[int] = []
        self.DNA3b: list[int] = []



