from utils import bits_to_int
from DNA import *

class Decoder:
    def __init__(self):
        pass

    def connection_based_single_DNA_one_chromosome_no_markers_no_grey(self, DNA: Single_DNA_one_chromosome):
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

            source_id = bits_to_int(source_bits) % 17
            target_id = bits_to_int(target_bits) % 17

            # Mapowanie wagi z 0-255 na -5.0 do 5.0
            raw_weight = bits_to_int(weight_bits)
            weight = (raw_weight / 255.0) * 10.0 - 5.0

            nn.append({
                'source': source_id,
                'target': target_id,
                'weight': weight
            })

        return nn

    def connection_based_single_DNA_one_chromosome_with_markers_no_grey(self, DNA: DNA):
        nn = []
        return nn