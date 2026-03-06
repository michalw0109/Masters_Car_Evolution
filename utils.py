import random

def binary_to_gray(n):
    """Konwertuje liczbę całkowitą na kod Graya."""
    return n ^ (n >> 1)

def gray_to_binary(n):
    """Konwertuje kod Graya z powrotem na liczbę całkowitą."""
    mask = n
    while mask != 0:
        mask >>= 1
        n ^= mask
    return n


def int_to_bits(n, num_bits):
    """Zamienia int na listę bitów [0, 1, ...]"""
    return [int(x) for x in bin(n)[2:].zfill(num_bits)]


def bits_to_int(bits):
    """Zamienia listę bitów na int."""
    bit_str = "".join(str(b) for b in bits)
    return int(bit_str, 2)

def generateRandomDna(length):
    return [random.choice([0, 1]) for _ in range(length)]


def connectionToDNA(connection):
    """
    Zamienia połążczenie w sieci z powrotem na czyste DNA.
    """
    dna = []

    # Zakoduj parametry
    dna.extend(int_to_bits(connection['source'], 8))
    dna.extend(int_to_bits(connection['target'], 8))

    # Odwróć mapowanie wagi
    norm_weight = int(((connection['weight'] + 5.0) / 10.0) * 255)
    norm_weight = max(0, min(255, norm_weight))  # clamp
    dna.extend(int_to_bits(norm_weight, 8))

    return dna