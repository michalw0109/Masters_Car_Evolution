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
        # wartości neuronów
        values = defaultdict(float)

        # ustaw wejścia
        for i, v in enumerate(input_vector):
            values[self.INPUTS[i]] = v

        # normalizacja połączeń: source < target
        normalized = []
        for c in connections:
            s, t, w = c["source"], c["target"], c["weight"]
            if s <= t:
                normalized.append((s, t, w))
            else:
                normalized.append((t, s, w))

        # sortowanie po target ID
        normalized.sort(key=lambda x: x[1])

        # propagacja
        for s, t, w in normalized:
            values[t] += values[s] * w

        # wyjścia
        return [values[i] for i in self.OUTPUTS]


    def connection_based_max_range_propagation(self, nn, input_vector):
        INPUTS = [0, 1, 2, 3, 4, 5]
        OUTPUTS = [251, 252, 253, 254]
        values = defaultdict(float)
        return [values[i] for i in OUTPUTS]

