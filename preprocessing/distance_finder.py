import numpy as np
import os
import re
from preprocessing.DistanceFinder import DistanceFinder

mask = len("preprocessing")
directory = os.path.dirname(os.path.abspath(__file__))[:-mask]+"/matrizes"

if __name__ == "__main__":

    finder = DistanceFinder()
    choices_map = {}

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file = open(directory+"/"+filename)
            lines = file.readlines()
            array = np.array(list(map(lambda line: list(map(int, line.strip('\n').split(","))), lines)))
            finder.set_g(array)
            key = re.sub(r'.*{', '{', filename)
            numb = filename.partition(":")[0]
            if key in choices_map:
                if finder.get_distance() > choices_map[key][0]:
                    choices_map[key] = (finder.get_distance(), numb)
            else:
                choices_map[key] = (finder.get_distance(), numb)
            print("Distance : " + str(finder.get_distance()) + " archive: " + filename)
            print("Matrix:")
            print(array)
            file.close()

    # Printing the files chosen
    print("Chosen matrices: ")
    choices = sorted(choices_map.values(), key=lambda choice: int(choice[1]))

    for choice in choices:
        print("File: " + choice[1] + " distance: " + str(choice[0]))
