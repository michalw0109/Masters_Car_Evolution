import math
from collections import defaultdict

class Computation:
    def __init__(self, inputs: list[int], outputs: list[int], marker=None):
        if marker is None:
            marker = [0, 1, 1, 1, 1, 1, 1, 1]

        self.INPUTS = inputs
        self.OUTPUTS = outputs
        self.MARKER = marker

        self.NR_OF_NEURONS = self.OUTPUTS[len(self.OUTPUTS) - 1] + 1

    def connection_based_sort_feed_forward(self, nn, input_vector):

        connections = nn
        values = defaultdict(float)
        inputs_set = set(self.INPUTS)

        # ustaw wejścia
        for i, v in enumerate(input_vector):
            values[self.INPUTS[i]] = v

        # zachowaj tylko krawędzie do przodu (source < target), pomijaj pętle i back-edges
        forward = []
        for c in connections:
            s, t, w = c["source"], c["target"], c["weight"]
            if s < t:
                forward.append((s, t, w))

        # sortowanie po target ID
        forward.sort(key=lambda x: x[1])

        # propagacja z aktywacją tanh per neuron po zsumowaniu wszystkich wejść
        i = 0
        while i < len(forward):
            t = forward[i][1]
            while i < len(forward) and forward[i][1] == t:
                s, _, w = forward[i]
                values[t] += values[s] * w
                i += 1
            if t not in inputs_set:
                values[t] = math.tanh(values[t])

        # wyjścia
        return [values[i] for i in self.OUTPUTS]


    def connection_based_max_range_propagation(self, nn, input_vector):
        INPUTS = [0, 1, 2, 3, 4, 5]
        OUTPUTS = [251, 252, 253, 254]
        values = defaultdict(float)
        return [values[i] for i in OUTPUTS]

