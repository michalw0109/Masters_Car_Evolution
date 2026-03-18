import random
from DNA import DNA
from Individual import Individual
from utils import *

class MutationCombiner:
    def __init__(self, mutations: list):
        self.mutations = mutations

    def mutate(self, child: Individual):
        for mutation in self.mutations:
            child = mutation(child)
        return child


class Mutation:
    def __init__(self, _MUTATION_RATE):
        self.MUTATION_RATE = _MUTATION_RATE

    def no_mutation(self, child: Individual):
        return child

    def random_bit_flip_single_DNA_one_chromosome(self, child: Individual):
        dna = child.dnaType.DNA
        for i in range(len(dna)):
            if random.random() < self.MUTATION_RATE:
                dna[i] = 1 - dna[i]
        child.dnaType.DNA = dna
        return child

    def random_insert_single_DNA_one_chromosome(self, child: Individual):

        dna = child.dnaType.DNA

        # Insert (Wstawienie losowego bitu/bloków)
        if random.random() < self.MUTATION_RATE:
            pos = random.randint(0, len(dna))
            # Wstawiamy mały fragment (może to być śmieć, a może fragment nowego genu)
            chunk = generateRandomDna(random.randint(10, 50))
            dna[pos:pos] = chunk

        child.dnaType.DNA = dna
        return child

    def random_delete_single_DNA_one_chromosome(self, child: Individual):

        dna = child.dnaType.DNA

        # Insert (Wstawienie losowego bitu/bloków)
        if random.random() < self.MUTATION_RATE:
            start = random.randint(0, len(dna))
            end = random.randint(0, len(dna))
            if end < start:
                (start, end) = (end, start)
            del dna[start:end]

        child.dnaType.DNA = dna
        return child

    def random_gene_copy_insert(self, child: Individual):
        return child



    def gene_duplication(self, dna, marker_seq):
        """
        Znajduje istniejący gen i kopiuje go.
        To potężny mechanizm - pozwala sieci mieć kopię zapasową
        funkcji, która potem może ewoluować w coś nowego.
        """
        if random.random() > self.MUTATION_RATE:
            return dna

        # Znajdź wszystkie indeksy markerów
        marker_indices = []
        for i in range(len(dna) - 8):
            if dna[i:i + 8] == marker_seq:
                marker_indices.append(i)

        if not marker_indices:
            return dna

        # Wybierz losowy gen do skopiowania (Marker + 24 bity danych)
        src_idx = random.choice(marker_indices)
        gene_block = dna[src_idx: src_idx + 32]

        # Wklej go w losowe miejsce (transpozycja)
        insert_pos = random.randint(0, len(dna))
        new_dna = dna[:]
        new_dna[insert_pos:insert_pos] = gene_block

        return new_dna