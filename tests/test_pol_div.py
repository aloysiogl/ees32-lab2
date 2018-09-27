from polynomial_codifiers.PolyDecoder import PolyDecoder
from polynomial_codifiers.PolyEncoder import PolyEncoder
import numpy as np


def test_division():
    num = np.array([0, 0, 0, 0, 1, 0, 1, 0], dtype=int)
    den = np.array([1, 0, 0], dtype=int)
    res = np.array([0, 0, 0, 0, 0, 0, 1, 1])
    encoder = PolyEncoder(num)

    polymult = np.mod(encoder.encode(den) + res, 2)

    print(polymult)

    print(PolyDecoder.pol_div(polymult, num))
    print(PolyDecoder.pol_div(polymult, den))


if __name__ == "__main__":
    test_division()
