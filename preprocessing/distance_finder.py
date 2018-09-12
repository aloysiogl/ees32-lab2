import numpy as np
import os
from preprocessing.DistanceFinder import DistanceFinder

directory = os.path.dirname(os.path.abspath(__file__))+"/matrizes"

if __name__ == "__main__":

    finder = DistanceFinder()

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file = open(directory+"/"+filename)
            lines = file.readlines()
            array = np.array(list(map(lambda line: list(map(int, line.strip('\n').split(","))), lines)))
            finder.set_g(array)
            print("Distance : " + str(finder.get_distance()) + " archive: " + filename)
            print("Matrix:")
            print(array)
            file.close()

