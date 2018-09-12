import numpy as np
from DistanceFinder import DistanceFinder

if __name__ == "__main__":
    dist = DistanceFinder()
    a = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                  [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                  [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                  [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]],)

    

    dist.set_g(a)
    print(dist.get_distance())
