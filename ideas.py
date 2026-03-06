import random
import numpy as np

class DNATools:
    @staticmethod
    def binary_to_gray(n):
        """Konwertuje liczbę całkowitą na kod Graya."""
        return n ^ (n >> 1)

    @staticmethod
    def gray_to_binary(n):
        """Konwertuje kod Graya z powrotem na liczbę całkowitą."""
        mask = n
        while mask != 0:
            mask >>= 1
            n ^= mask
        return n

    @staticmethod
    def int_to_bits(n, num_bits):
        """Zamienia int na listę bitów [0, 1, ...]. Zastosowanie Gray Code."""
        gray = DNATools.binary_to_gray(n)
        return [int(x) for x in bin(gray)[2:].zfill(num_bits)]

    @staticmethod
    def bits_to_int(bits):
        """Zamienia listę bitów na int (dekodując z Gray Code)."""
        bit_str = "".join(str(b) for b in bits)
        gray = int(bit_str, 2)
        return DNATools.gray_to_binary(gray)

    @staticmethod
    def generate_random_dna(length):
        return [random.choice([0, 1]) for _ in range(length)]


class GeneticTranslator:
    def __init__(self):
        # Marker: 11110000 - unikalna sekwencja startowa genu
        self.MARKER = [1, 1, 1, 1, 0, 0, 0, 0]
        self.GENE_SIZE = 24  # 8 source + 8 target + 8 weight = 24 bity danych (+ marker)

    def dna_to_network(self, dna):
        """
        Skanuje DNA (lista bitów) i zwraca strukturę sieci (lista połączeń).
        Zwraca listę słowników: [{'source': int, 'target': int, 'weight': float}, ...]
        """
        connections = []
        i = 0
        limit = len(dna) - len(self.MARKER) - self.GENE_SIZE

        while i < limit:
            # Sprawdź czy tutaj jest marker
            if dna[i: i + 8] == self.MARKER:
                # Znaleziono gen! Przeskakujemy marker
                idx = i + 8

                # Dekodowanie
                source_bits = dna[idx: idx + 8]
                target_bits = dna[idx + 8: idx + 16]
                weight_bits = dna[idx + 16: idx + 24]

                source_id = DNATools.bits_to_int(source_bits)
                target_id = DNATools.bits_to_int(target_bits)

                # Mapowanie wagi z 0-255 na -5.0 do 5.0
                raw_weight = DNATools.bits_to_int(weight_bits)
                weight = (raw_weight / 255.0) * 10.0 - 5.0

                connections.append({
                    'source': source_id,
                    'target': target_id,
                    'weight': weight
                })

                # Przeskocz przeczytany gen
                i += 8 + self.GENE_SIZE
            else:
                # To nie marker, idziemy bit dalej (to jest junk dna)
                i += 1

        return connections

    def network_to_dna(self, connections):
        """
        Zamienia strukturę sieci z powrotem na czyste DNA.
        Można dodać losowe "śmieci" między genami dla realizmu.
        """
        dna = []
        for conn in connections:
            # Dodaj Marker
            dna.extend(self.MARKER)

            # Zakoduj parametry
            dna.extend(DNATools.int_to_bits(conn['source'], 8))
            dna.extend(DNATools.int_to_bits(conn['target'], 8))

            # Odwróć mapowanie wagi
            norm_weight = int(((conn['weight'] + 5.0) / 10.0) * 255)
            norm_weight = max(0, min(255, norm_weight))  # clamp
            dna.extend(DNATools.int_to_bits(norm_weight, 8))

            # Opcjonalnie: dodaj trochę śmieciowego DNA między genami (Junk DNA)
            # Pomaga to chronić geny przed zniszczeniem przy krzyżowaniu w jednym punkcie
            junk_size = random.randint(0, 5)
            dna.extend(DNATools.generate_random_dna(junk_size))

        return dna


class Mutator:
    def __init__(self, mutation_rate=0.01):
        self.rate = mutation_rate

    def point_mutation(self, dna):
        """Standardowa mutacja: każdy bit ma szansę na zmianę."""
        new_dna = dna[:]
        for i in range(len(new_dna)):
            if random.random() < self.rate:
                new_dna[i] = 1 - new_dna[i]  # Flip bit
        return new_dna

    def indel_mutation(self, dna):
        """
        Insertion/Deletion. Zmienia długość DNA!
        Kluczowe dla ewolucji nowych funkcji (zmiennej topologii).
        """
        new_dna = dna[:]

        # Insert (Wstawienie losowego bitu/bloków)
        if random.random() < self.rate:
            pos = random.randint(0, len(new_dna))
            # Wstawiamy mały fragment (może to być śmieć, a może fragment nowego genu)
            chunk = DNATools.generate_random_dna(random.randint(1, 10))
            new_dna[pos:pos] = chunk

        # Delete (Usunięcie fragmentu)
        if random.random() < self.rate and len(new_dna) > 50:
            start = random.randint(0, len(new_dna) - 10)
            end = start + random.randint(1, 10)
            del new_dna[start:end]

        return new_dna

    def gene_duplication(self, dna, marker_seq):
        """
        Znajduje istniejący gen i kopiuje go.
        To potężny mechanizm - pozwala sieci mieć kopię zapasową
        funkcji, która potem może ewoluować w coś nowego.
        """
        if random.random() > self.rate:
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


class Crossover:
    @staticmethod
    def homologous_crossover(dna_a, dna_b, marker_seq):
        """
        Próbuje krzyżować w miejscach "bezpiecznych", czyli między genami
        lub wyrównując geny o podobnej strukturze.

        Tutaj zastosujemy uproszczone podejście: "Cut and Splice"
        ale tylko w punktach, gdzie w OBU łańcuchach występuje Marker (lub śmieci).
        """
        # Znajdź punkty startu markerów w obu DNA
        points_a = [i for i in range(len(dna_a) - 8) if dna_a[i:i + 8] == marker_seq]
        points_b = [i for i in range(len(dna_b) - 8) if dna_b[i:i + 8] == marker_seq]

        # Jeśli któryś nie ma genów, zwróć dłuższego
        if not points_a or not points_b:
            return dna_a if len(dna_a) > len(dna_b) else dna_b

        # Wybierz losowy punkt cięcia z dostępnych markerów (wyrównanie do genu)
        # To symuluje synapsis chromosomów
        cut_a = random.choice(points_a)

        # Znajdź punkt w B, który jest "blisko" punktu A (relatywnie)
        # Normalizujemy pozycję (0.0 - 1.0)
        rel_pos = cut_a / len(dna_a)
        target_b_idx = int(rel_pos * len(points_b))
        cut_b = points_b[target_b_idx]

        # Stwórz dziecko: Poczatek z A, Koniec z B
        child = dna_a[:cut_a] + dna_b[cut_b:]
        return child


class DiploidOrganism:
    def __init__(self, dna1, dna2):
        self.chromosome_a = dna1
        self.chromosome_b = dna2
        self.translator = GeneticTranslator()

    def express_phenotype(self):
        """
        Tworzy sieć z dwóch nici DNA.
        Logika dominacji:
        1. Dekodujemy oba łańcuchy.
        2. Scalamy połączenia.
        """
        net_a = self.translator.dna_to_network(self.chromosome_a)
        net_b = self.translator.dna_to_network(self.chromosome_b)

        final_network = {}  # Używamy dict (src, tgt) -> weight dla łatwego scalania

        # Funkcja pomocnicza do dodawania połączeń
        def add_conns(conns):
            for c in conns:
                key = (c['source'], c['target'])
                if key in final_network:
                    # Jeśli połączenie już istnieje (z drugiego chromosomu), uśredniamy
                    # To symuluje kodominację
                    final_network[key] = (final_network[key] + c['weight']) / 2
                else:
                    final_network[key] = c['weight']

        add_conns(net_a)
        add_conns(net_b)

        # Zamiana z powrotem na listę dla autka
        return [{'source': k[0], 'target': k[1], 'weight': v} for k, v in final_network.items()]

    @staticmethod
    def sexual_reproduction(parent1, parent2):
        """
        Mieszanie diploidalne (Mejoza).
        Dziecko dostaje jeden chromosom od P1 i jeden od P2.
        NIE tniemy DNA! Dzięki temu geny się nie psują.
        """
        # Gameta rodzica 1 (losowy wybór jednego chromosomu)
        gamete_1 = parent1.chromosome_a if random.random() < 0.5 else parent1.chromosome_b

        # Gameta rodzica 2
        gamete_2 = parent2.chromosome_a if random.random() < 0.5 else parent2.chromosome_b

        # Możemy tu dodać mutację do gamet
        mutator = Mutator()
        gamete_1 = mutator.point_mutation(gamete_1)  # Lekka mutacja
        # gamete_1 = mutator.indel_mutation(gamete_1) # Indele tylko czasami

        return DiploidOrganism(gamete_1, gamete_2)


class AdvancedNetworkDecoder:
    def __init__(self):
        self.INPUTS = 5  # ID: 0, 1, 2, 3, 4
        self.OUTPUTS = 4  # ID: 5, 6, 7, 8
        # Wszystko od 9 w górę to neurony ukryte

        # Marker i rozmiar genu pozostają bez zmian
        self.MARKER = [1, 1, 1, 1, 0, 0, 0, 0]

    def get_node_type(self, node_id):
        """Pomocnicza funkcja określająca typ neuronu po ID."""
        if node_id < self.INPUTS:
            return "SENSOR"
        elif node_id < (self.INPUTS + self.OUTPUTS):
            return "ACTION"
        else:
            return "HIDDEN"

    def dna_to_network(self, dna):
        connections = []
        hidden_nodes_present = set()  # Śledzimy, które ukryte neurony są w użyciu

        i = 0
        limit = len(dna) - 32  # Marker (8) + Gen (24)

        while i < limit:
            # Szukamy markera
            if dna[i: i + 8] == self.MARKER:
                idx = i + 8

                # Pobieramy surowe 8-bitowe wartości (0-255)
                # Używamy wcześniej zdefiniowanych narzędzi do konwersji bitów
                raw_source = DNATools.bits_to_int(dna[idx: idx + 8])
                raw_target = DNATools.bits_to_int(dna[idx + 8: idx + 16])
                raw_weight = DNATools.bits_to_int(dna[idx + 16: idx + 24])

                # --- LOGIKA MAPOWANIA (TUTAJ JEST ZMIANA) ---

                # 1. Źródło (Source): Może być Sensorem lub Ukrytym (rzadziej Akcją - rekurencja)
                # Ograniczamy przestrzeń, np. do 64 neuronów, żeby sieć nie była za rzadka na początku
                # Ale ewolucja może to zwiększyć. Dla uproszczenia weźmy modulo 32.
                MAX_NODES = 32
                source_id = raw_source % MAX_NODES

                # 2. Cel (Target): NIE MOŻE być Sensorem. Musi być Akcją lub Ukrytym.
                # Matematyczna sztuczka: mapujemy 0-255 na zakres [5 - 31]
                # (Omijamy zakres 0-4)
                possible_targets = MAX_NODES - self.INPUTS  # 32 - 5 = 27
                target_id = (raw_target % possible_targets) + self.INPUTS

                # 3. Waga (-5.0 do 5.0)
                weight = (raw_weight / 255.0) * 10.0 - 5.0

                # --- Walidacja ---
                # Nie chcemy pętli do samego siebie (chyba że chcesz rekurencję)
                if source_id != target_id:
                    connections.append({
                        'source': source_id,
                        'target': target_id,
                        'weight': weight
                    })

                    # Rejestrujemy, że te neurony istnieją
                    if self.get_node_type(source_id) == "HIDDEN":
                        hidden_nodes_present.add(source_id)
                    if self.get_node_type(target_id) == "HIDDEN":
                        hidden_nodes_present.add(target_id)

                i += 32  # Przeskok
            else:
                i += 1

        return connections, hidden_nodes_present


import math


class NeuralNetwork:
    def __init__(self, connections):
        self.connections = connections
        # Mapa wartości neuronów: {node_id: value}
        # Inicjalizujemy zerami. Ważne, żeby wyczyścić przed każdym krokiem symulacji (lub nie, jeśli ma pamięć)
        self.node_values = {}

    def activate(self, inputs):
        """
        inputs: lista 5 wartości z sensorów [d1, d2, d3, d4, d5]
        Zwraca: lista 4 wartości dla silników [a1, a2, a3, a4]
        """
        # 1. Wpisz wartości sensorów do mapy (ID 0-4)
        for i, val in enumerate(inputs):
            self.node_values[i] = val

        # 2. Wyzeruj resztę (akcje i ukryte), chyba że chcesz pamięć (Recurrent Neural Network)
        # Tutaj wersja bez pamięci (reset co klatkę):
        # Znajdź wszystkie unikalne nody z połączeń
        all_nodes = set()
        for c in self.connections:
            all_nodes.add(c['source'])
            all_nodes.add(c['target'])

        # Inicjalizuj zerem te, które nie są sensorami
        for node in all_nodes:
            if node >= 5:
                self.node_values[node] = 0.0

        # 3. Propagacja (Feed Forward)
        # Ponieważ topologia jest chaotyczna, musimy posortować topologicznie LUB
        # po prostu przeiterować połączenia kilka razy, aby sygnał przeszedł przez warstwy ukryte.
        # W prostych symulacjach wystarczy 1-3 przejścia pętli.

        steps = 3  # Pozwala sygnałowi przejść Sensor -> Hidden -> Hidden -> Action

        # Słownik tymczasowy na nowo obliczone wartości w tym kroku
        # Żeby nie nadpisywać w trakcie iteracji (synchroniczna aktualizacja)
        for _ in range(steps):
            next_values = {k: 0.0 for k in all_nodes if k >= 5}  # Zeruj sumy

            for conn in self.connections:
                src = conn['source']
                tgt = conn['target']
                w = conn['weight']

                # Jeśli źródło ma wartość (z poprzedniego kroku lub inputu)
                if src in self.node_values:
                    # Dodajemy ważony sygnał
                    val_to_add = self.node_values[src] * w

                    # Target musi być > 4 (nie nadpisujemy sensorów!)
                    if tgt >= 5:
                        next_values[tgt] += val_to_add

            # Aplikacja funkcji aktywacji (np. Tanh lub Sigmoid) i aktualizacja stanu
            for node_id, sum_val in next_values.items():
                # Funkcja aktywacji: Tanh (daje -1 do 1) - dobra do sterowania (lewo/prawo)
                self.node_values[node_id] = math.tanh(sum_val)

        # 4. Pobierz wyniki z outputów (ID 5-8)
        outputs = []
        for i in range(5, 9):
            outputs.append(self.node_values.get(i, 0.0))  # 0.0 jeśli nic nie podłączone

        return outputs


class FeedForwardDecoder:
    def __init__(self):
        # Konfiguracja adresów
        self.INPUT_IDS = range(0, 5)  # 0, 1, 2, 3, 4
        self.OUTPUT_IDS = range(252, 256)  # 252, 253, 254, 255
        # Wszystko pomiędzy (5-251) to Hidden

        self.MARKER = [1, 1, 1, 1, 0, 0, 0, 0]

    def dna_to_feedforward_network(self, dna):
        connections = []
        active_nodes = set()  # Zbiór aktywnych neuronów ukrytych

        i = 0
        limit = len(dna) - 32

        while i < limit:
            if dna[i: i + 8] == self.MARKER:
                idx = i + 8

                # Pobierz surowe dane (0-255)
                raw_a = DNATools.bits_to_int(dna[idx: idx + 8])
                raw_b = DNATools.bits_to_int(dna[idx + 8: idx + 16])
                raw_w = DNATools.bits_to_int(dna[idx + 16: idx + 24])

                # --- TWOJA LOGIKA: SORTOWANIE ID ---
                # To gwarantuje brak pętli zwrotnych!
                # Sygnał zawsze płynie od mniejszego do większego ID.
                source = min(raw_a, raw_b)
                target = max(raw_a, raw_b)

                weight = (raw_w / 255.0) * 10.0 - 5.0

                # --- FILTROWANIE ---
                valid = True

                # 1. Nie łączymy tego samego ze sobą (pętla własna)
                if source == target: valid = False

                # 2. Outputy nie mogą być źródłem (są na końcu łańcucha pokarmowego)
                if source in self.OUTPUT_IDS: valid = False

                # 3. Inputy nie mogą być celem (są na początku)
                if target in self.INPUT_IDS: valid = False

                if valid:
                    connections.append({
                        'source': source,
                        'target': target,
                        'weight': weight
                    })

                    # Rejestrujemy użyte hiddeny (żeby wiedzieć, które obliczać)
                    if source not in self.INPUT_IDS and source not in self.OUTPUT_IDS:
                        active_nodes.add(source)
                    if target not in self.INPUT_IDS and target not in self.OUTPUT_IDS:
                        active_nodes.add(target)

                i += 32
            else:
                i += 1

        # Sortujemy połączenia po Target ID.
        # To optymalizacja dla obliczeń: najpierw liczymy to, co wchodzi do wcześniejszych neuronów.
        connections.sort(key=lambda x: x['target'])

        return connections, sorted(list(active_nodes))


import math


class FastNetworkRunner:
    def __init__(self, connections, active_hidden_nodes):
        self.connections = connections
        # Mamy posortowaną listę aktywnych neuronów ukrytych
        self.hidden_nodes = active_hidden_nodes
        # Mapa wartości (resetowana co klatkę)
        self.values = {}

    def run(self, inputs):
        """
        inputs: Lista 5 wartości float (np. odległości z sensorów)
        zwraca: Lista 4 wartości float (sterowanie)
        """

        # 1. Wpisz Inputy (ID 0-4)
        for i, val in enumerate(inputs):
            self.values[i] = val

        # 2. Wyzeruj Hidden i Output (ID 5-255)
        # Wystarczy wyzerować te, które są faktycznie używane w sieci
        # (Optymalizacja: nie zerujemy tablicy 256 elementowej, tylko słownik)
        for node_id in self.hidden_nodes:
            self.values[node_id] = 0.0
        # Outputy (ID 252-255)
        for i in range(252, 256):
            self.values[i] = 0.0

        # 3. Propagacja (Jedno przejście!)
        # Ponieważ w Decoderze posortowaliśmy connection po 'target',
        # a ID rosną, możemy po prostu iterować po liście połączeń.
        # Ale bezpieczniej i czytelniej jest iterować po WĘZŁACH.

        # Ponieważ struktura jest rzadka (sparse), iterujemy po liście połączeń.
        # Ważne: Wymagamy, aby połączenia były przetwarzane w takiej kolejności,
        # że źródło jest już obliczone.
        # W naszym systemie Source < Target. Więc jeśli mamy listę połączeń i
        # po prostu będziemy brać wartości ze 'source', to zadziała, pod warunkiem,
        # że 'source' jest już gotowy.

        # Podejście najpewniejsze (Topology Walk):
        # Ponieważ source < target, idziemy po ID neuronów rosnąco.

        # Najpierw obliczmy wpływy. Grupujemy połączenia wg Targetu.
        # (To można zrobić raz w konstruktorze dla wydajności!)

        pass
        # (Poniżej zoptymalizowana wersja run_optimized)

    def run_optimized(self, inputs):
        # 1. Inputy
        state = {i: v for i, v in enumerate(inputs)}

        # 2. Przeliczamy sieć
        # Ponieważ connections są posortowane wg Targetu (w Decoderze),
        # a my musimy sumować wpływy dla danego Targetu.

        for conn in self.connections:
            src = conn['source']
            tgt = conn['target']
            w = conn['weight']

            # Pobierz wartość źródła (jeśli nie ma, to 0.0 - np. niepodłączony hidden)
            src_val = state.get(src, 0.0)

            # Dodaj do celu
            if tgt not in state:
                state[tgt] = 0.0
            state[tgt] += src_val * w

            # UWAGA: Tu jest haczyk.
            # Musimy zaaplikować funkcję aktywacji (Tanh/Relu) na neuronie hidden
            # ZANIM zostanie on użyty jako źródło dla kogoś innego.
            # Dlatego prosta pętla po connections nie zadziała idealnie, jeśli
            # nie są posortowane topologicznie idealnie.

            # Rozwiązanie "Leniwe" (Lazy Evaluation) lub "Warstwowe":
            # Skoro ID rosną (0 -> 255), iterujmy po ID.
            pass


class LinearTopologyRunner:
    def __init__(self, connections):
        # Grupujemy połączenia WCHODZĄCE do danego noda
        # structure[target_id] = [(source_id, weight), (source_id, weight)...]
        self.structure = {}
        self.sorted_targets = set()

        for c in connections:
            t = c['target']
            if t not in self.structure:
                self.structure[t] = []
                self.sorted_targets.add(t)
            self.structure[t].append((c['source'], c['weight']))

        self.sorted_targets = sorted(list(self.sorted_targets))

    def predict(self, inputs):
        # Stan pamięci dla całej sieci
        # Inputy (0-4) są już gotowe
        memory = {i: val for i, val in enumerate(inputs)}

        # Iterujemy po targetach od najmniejszego do największego (5...255)
        # To jest KLUCZ. Ponieważ Source < Target, to gdy liczymy Target=10,
        # wiemy że wszystkie Source (np. 2, 5, 8) są już obliczone w tej pętli wcześniej!
        for node_id in self.sorted_targets:
            total_input = 0.0

            # Sumuj wejścia
            for src_id, weight in self.structure[node_id]:
                # Pobierz wartość źródła (może być inputem lub hiddenem obliczonym przed chwilą)
                val = memory.get(src_id, 0.0)
                total_input += val * weight

            # Aktywacja
            # Dla Outputów (252-255) też dajemy Tanh lub Sigmoid, albo liniową
            memory[node_id] = math.tanh(total_input)

        # Zwróć tylko Outputy
        return [memory.get(i, 0.0) for i in range(252, 256)]