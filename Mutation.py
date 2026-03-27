import random
from DNA import DNA
from Individual import Individual
from utils import *
from copy import deepcopy

class MutationCombiner:
    def __init__(self, mutations: list):
        self.mutations = mutations

    def mutate(self, child: Individual):
        for mutation in self.mutations:
            child = mutation(child)
        return child


class Mutation:
    def __init__(self):
        pass

    def no_mutation(self, child: Individual):
        return child

    class mutate_single_DNA_one_chromosome:
        """
        Intelligent mutations that understand each DNA encoding.
        Each method receives an Individual, mutates it in-place on a deepcopy,
        and returns the (possibly mutated) Individual.
        """

        def __init__(self, _MUTATION_RATE, _MARKER=None):
            self.MUTATION_RATE = _MUTATION_RATE
            if _MARKER is None:
                _MARKER = [0, 1, 1, 1, 1, 1, 1, 1]
            self.MARKER = _MARKER

        # ── internal helpers ──────────────────────────────────────────────────

        @staticmethod
        def _int_to_bits(value, n=8):
            return [(value >> (n - 1 - b)) & 1 for b in range(n)]



        def random_bit_flip(self, child: Individual):
            dna = child.dnaType.DNA
            for i in range(len(dna)):
                if random.random() < self.MUTATION_RATE:
                    dna[i] = 1 - dna[i]
            child.dnaType.DNA = dna
            return child

        def random_insert(self, child: Individual):

            dna = child.dnaType.DNA

            # Insert (Wstawienie losowego bitu/bloków)
            if random.random() < self.MUTATION_RATE:
                pos = random.randint(0, len(dna))
                # Wstawiamy mały fragment (może to być śmieć, a może fragment nowego genu)
                chunk = generateRandomDna(random.randint(10, 50))
                dna[pos:pos] = chunk

            child.dnaType.DNA = dna
            return child

        def random_delete(self, child: Individual):

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





        # =========================================================================
        # Smart structure-aware mutations (mirror of Crossover / Decoder classes)
        # =========================================================================



        # ─────────────────────────────────────────────────────────────────────
        # 1. connection_based  (gene = 24 bits: 8 source | 8 target | 8 weight)
        # ─────────────────────────────────────────────────────────────────────

        def connection_based(self, individual: Individual):
            """
            Gene-aware mutations on 24-bit connection genes.
            Operations (one chosen per call when mutation fires):
              duplicate  – copy a random 24-bit gene and insert it elsewhere
              delete     – remove a random 24-bit gene
            """
            ind = deepcopy(individual)
            if random.random() >= self.MUTATION_RATE:
                return ind

            GENE = 24
            dna = ind.dnaType.DNA
            n_genes = len(dna) // GENE
            if n_genes == 0:
                return ind

            new_dna = list(dna)
            op = random.choice(['duplicate', 'delete'])

            if op == 'duplicate':
                src = random.randint(0, n_genes - 1) * GENE
                gene = new_dna[src:src + GENE]
                dst = random.randint(0, len(new_dna) // GENE) * GENE
                new_dna = new_dna[:dst] + gene + new_dna[dst:]

            elif op == 'delete' and n_genes > 1:
                src = random.randint(0, n_genes - 1) * GENE
                new_dna = new_dna[:src] + new_dna[src + GENE:]

            ind.dnaType.DNA = new_dna
            return ind

        # ─────────────────────────────────────────────────────────────────────
        # 2. connection_based_markers
        #    gene = MARKER(8) | source(8) | target(8) | weight(8) = 32 bits
        #    flanked by arbitrary junk DNA
        # ─────────────────────────────────────────────────────────────────────

        def connection_based_markers(self, individual: Individual):
            """
            Scans the DNA for valid marker-prefixed genes and performs:
              duplicate     – copy a gene (with marker) to another position
              delete        – remove a gene (and its marker)
            """
            ind = deepcopy(individual)
            if random.random() >= self.MUTATION_RATE:
                return ind

            MARKER = self.MARKER
            ML = len(MARKER)   # 8
            GENE = ML + 24     # 32

            dna = list(ind.dnaType.DNA)

            # Locate all gene starts
            gene_pos = []
            i = 0
            while i <= len(dna) - GENE:
                if dna[i:i + ML] == MARKER:
                    gene_pos.append(i)
                    i += GENE
                else:
                    i += 1

            op = random.choice(['duplicate', 'delete'])

            if op == 'duplicate' and gene_pos:
                pos = random.choice(gene_pos)
                gene = dna[pos:pos + GENE]
                dst = random.randint(0, len(dna))
                dna = dna[:dst] + gene + dna[dst:]

            elif op == 'delete' and len(gene_pos) > 1:
                pos = random.choice(gene_pos)
                dna = dna[:pos] + dna[pos + GENE:]


            ind.dnaType.DNA = dna
            return ind

        # ─────────────────────────────────────────────────────────────────────
        # 3. matrix_connections
        #    DNA[0:256]  = 16×16 adjacency matrix (1 bit per cell)
        #    DNA[256:]   = weight bytes (8 bits each, one per active connection)
        # ─────────────────────────────────────────────────────────────────────

        def matrix_connections(self, individual: Individual):
            """
            Mutations aware of the 256-bit matrix header + weight tail.
            Operations:
              weight_duplicate   – copy one weight byte to another slot in the tail
              weight_delete      – remove one weight byte from the tail
            """
            ind = deepcopy(individual)
            if random.random() >= self.MUTATION_RATE:
                return ind

            dna = list(ind.dnaType.DNA)
            if len(dna) < 256:
                return ind

            op = random.choice(['weight_duplicate', 'weight_delete'])


            if op == 'weight_duplicate' and len(dna) > 264:
                tail = dna[256:]
                nw = len(tail) // 8
                if nw:
                    i = random.randint(0, nw - 1) * 8
                    block = tail[i:i + 8]
                    dst = random.randint(0, nw) * 8
                    new_tail = tail[:dst] + block + tail[dst:]
                    dna = dna[:256] + new_tail

            elif op == 'weight_delete' and len(dna) > 264:
                tail = dna[256:]
                nw = len(tail) // 8
                if nw > 1:
                    i = random.randint(0, nw - 1) * 8
                    dna = dna[:256] + tail[:i] + tail[i + 8:]

            ind.dnaType.DNA = dna
            return ind

        # ─────────────────────────────────────────────────────────────────────
        # 4. triangular_matrix_connections
        #    DNA[0:120]  = upper-triangular 16×16 matrix (120 bits)
        #    DNA[120:]   = weight bytes (8 bits each)
        # ─────────────────────────────────────────────────────────────────────

        def triangular_matrix_connections(self, individual: Individual):
            """
            Same spirit as matrix_connections but for the 120-bit upper-triangular header.
            Operations:
              weight_duplicate   – copy one weight byte to another slot
              weight_delete      – remove one weight byte
            """
            ind = deepcopy(individual)
            if random.random() >= self.MUTATION_RATE:
                return ind

            dna = list(ind.dnaType.DNA)
            if len(dna) < 120:
                return ind

            op = random.choice(['weight_duplicate', 'weight_delete'])


            if op == 'weight_duplicate' and len(dna) > 128:
                tail = dna[120:]
                nw = len(tail) // 8
                if nw:
                    i = random.randint(0, nw - 1) * 8
                    block = tail[i:i + 8]
                    dst = random.randint(0, nw) * 8
                    new_tail = tail[:dst] + block + tail[dst:]
                    dna = dna[:120] + new_tail

            elif op == 'weight_delete' and len(dna) > 128:
                tail = dna[120:]
                nw = len(tail) // 8
                if nw > 1:
                    i = random.randint(0, nw - 1) * 8
                    dna = dna[:120] + tail[:i] + tail[i + 8:]

            ind.dnaType.DNA = dna
            return ind

        # ─────────────────────────────────────────────────────────────────────
        # 5. fixed_topology  (DNA = weights only, 8 bits each, fixed count)
        # ─────────────────────────────────────────────────────────────────────

        def fixed_topology(self, individual: Individual):
            """
            Topology is fixed, so only the weight bytes matter.
            Operations:
              weight_shift   – nudge a weight byte by a small delta
              weight_copy    – copy one weight byte to another position
              weight_flip    – replace a weight byte with a random value
              weight_negate  – flip the weight from positive to negative range
            """
            ind = deepcopy(individual)
            if random.random() >= self.MUTATION_RATE:
                return ind

            dna = list(ind.dnaType.DNA)
            nw = len(dna) // 8
            if nw == 0:
                return ind

            op = random.choice(['weight_copy', 'weight_delete'])
            i = random.randint(0, nw - 1) * 8



            if op == 'weight_copy' and nw > 1:
                src = random.randint(0, nw - 1) * 8
                dna[i:i + 8] = dna[src:src + 8]

            elif op == 'weight_delete' and len(dna) > 128:
                dna = dna[:i] + dna[i + 8:]


            ind.dnaType.DNA = dna
            return ind

        # ─────────────────────────────────────────────────────────────────────
        # 6. grammar_matrix
        #    DNA[0:96]    = 8 grammar rules × 12 bits
        #    DNA[96:224]  = 8 final rules    × 16 bits
        #    DNA[224:]    = weight tail       × 8 bits each
        # ─────────────────────────────────────────────────────────────────────

        def grammar_matrix(self, individual: Individual):
            """
            Mutations aware of the grammar rule structure.
            Operations:
              swap_grammar_rules    – exchange two 12-bit grammar rules
              duplicate_grammar_rule – overwrite one grammar rule with another
              flip_grammar_bit      – flip one bit inside a grammar rule
              swap_final_rules      – exchange two 16-bit final rules
              duplicate_final_rule  – overwrite one final rule with another
              flip_final_bit        – flip one bit inside a final rule
              weight_shift          – nudge one weight byte in the tail
              weight_duplicate      – copy one weight byte in the tail
              weight_delete         – remove one weight byte from the tail
            """
            ind = deepcopy(individual)
            if random.random() >= self.MUTATION_RATE:
                return ind

            dna = list(ind.dnaType.DNA)
            if len(dna) < 224:
                return ind

            op = random.choice([
                'swap_grammar_rules', 'duplicate_grammar_rule',
                'swap_final_rules', 'duplicate_final_rule',
                'weight_duplicate', 'weight_delete'
            ])

            if op == 'swap_grammar_rules':
                a, b = random.sample(range(8), 2)
                sa, sb = a * 12, b * 12
                dna[sa:sa + 12], dna[sb:sb + 12] = dna[sb:sb + 12], dna[sa:sa + 12]

            elif op == 'duplicate_grammar_rule':
                src, dst = random.sample(range(8), 2)
                dna[dst * 12:(dst + 1) * 12] = dna[src * 12:(src + 1) * 12]


            elif op == 'swap_final_rules':
                a, b = random.sample(range(8), 2)
                sa, sb = 96 + a * 16, 96 + b * 16
                dna[sa:sa + 16], dna[sb:sb + 16] = dna[sb:sb + 16], dna[sa:sa + 16]

            elif op == 'duplicate_final_rule':
                src, dst = random.sample(range(8), 2)
                dna[96 + dst * 16:96 + (dst + 1) * 16] = dna[96 + src * 16:96 + (src + 1) * 16]


            elif op == 'weight_duplicate' and len(dna) > 232:
                tail = dna[224:]
                nw = len(tail) // 8
                if nw:
                    i = random.randint(0, nw - 1) * 8
                    block = tail[i:i + 8]
                    dst = random.randint(0, nw) * 8
                    new_tail = tail[:dst] + block + tail[dst:]
                    dna = dna[:224] + new_tail

            elif op == 'weight_delete' and len(dna) > 232:
                tail = dna[224:]
                nw = len(tail) // 8
                if nw > 1:
                    i = random.randint(0, nw - 1) * 8
                    dna = dna[:224] + tail[:i] + tail[i + 8:]

            ind.dnaType.DNA = dna
            return ind

        # ─────────────────────────────────────────────────────────────────────
        # 7. cellular_division
        #    Each instruction = 25 bits: opcode(1) | data(24)
        #    opcode 0 → add cell       (8-bit cell_id | 16-bit pad)
        #    opcode 1 → add connection (8-bit src | 8-bit dst | 8-bit weight)
        # ─────────────────────────────────────────────────────────────────────

        def cellular_division(self, individual: Individual):
            """
            Mutations aware of the 25-bit instruction structure.
            Operations:
              duplicate_instruction  – copy any instruction and insert it
              delete_instruction     – remove any instruction

            """
            ind = deepcopy(individual)
            if random.random() >= self.MUTATION_RATE:
                return ind

            INSTR = 25
            dna = list(ind.dnaType.DNA)
            n = len(dna) // INSTR
            if n == 0:
                return ind

            op = random.choice([
                'duplicate_instruction', 'delete_instruction'
            ])

            if op == 'duplicate_instruction':
                src = random.randint(0, n - 1) * INSTR
                instr = dna[src:src + INSTR]
                dst = random.randint(0, len(dna) // INSTR) * INSTR
                dna = dna[:dst] + instr + dna[dst:]

            elif op == 'delete_instruction' and n > 1:
                src = random.randint(0, n - 1) * INSTR
                dna = dna[:src] + dna[src + INSTR:]


            ind.dnaType.DNA = dna
            return ind

    class mutate_double_DNA_one_chromosome:
        """
        Intelligent mutations that understand each DNA encoding.
        Each method receives an Individual, mutates it in-place on a deepcopy,
        and returns the (possibly mutated) Individual.
        """

        def __init__(self, _MUTATION_RATE, _MARKER=None):
            self.MUTATION_RATE = _MUTATION_RATE
            if _MARKER is None:
                _MARKER = [0, 1, 1, 1, 1, 1, 1, 1]
            self.MARKER = _MARKER

        # ── internal helpers ──────────────────────────────────────────────────

        @staticmethod
        def _int_to_bits(value, n=8):
            return [(value >> (n - 1 - b)) & 1 for b in range(n)]



        def random_bit_flip(self, child: Individual):
            return child