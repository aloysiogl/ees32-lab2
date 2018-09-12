import numpy as np

class DistanceFinder:
    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.G = None
        self.k = None
        self.n = None
        self.distance = None

    def set_g(self, g):
        """Sets the G matix"""
        self.k, self.n = g.shape
        self.__calculate_distance()

    def __calculate_distance(self):
        for i in range(2**self.k):
            inp = [(i >> x) % 2 for x in range(self.k)]
            print(inp)
            self.inputs.append([])

    def get_distance(self):
        return self.distance