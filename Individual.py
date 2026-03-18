from DNA import DNA

class Individual:
    def __init__(self):
        self.dnaType = None
        self.nn = None
        self.fitness = 0
        self.testFitness = 0
        self.new = True

    def sortKey(self):
        return self.fitness
