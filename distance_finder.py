import numpy as np
from DistanceFinder import DistanceFinder

if __name__ == "__main__":
    dist = DistanceFinder()
    a = np.array([[1,1],
                  [1,3],
                  [1,4]],)
    dist.set_g(a)