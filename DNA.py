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



