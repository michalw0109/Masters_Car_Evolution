from collections import defaultdict

class Computation:
    def __init__(self):
        pass

    def connection_based_sort_feed_forward(self, nn, input_vector):

        INPUTS = [0, 1, 2, 3, 4, 5]
        OUTPUTS = [12, 13, 14, 15]

        connections = nn
        # wartości neuronów
        values = defaultdict(float)

        # ustaw wejścia
        for i, v in enumerate(input_vector):
            values[INPUTS[i]] = v

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
        return [values[i] for i in OUTPUTS]


    def connection_based_max_range_propagation(self, nn, input_vector):
        INPUTS = [0, 1, 2, 3, 4, 5]
        OUTPUTS = [251, 252, 253, 254]
        values = defaultdict(float)
        return [values[i] for i in OUTPUTS]

