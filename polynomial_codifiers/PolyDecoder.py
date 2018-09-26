import numpy as np
from polynomial_codifiers.PolyEncoder import PolyEncoder


class PolyDecoder:
    def __init__(self, g):
        self.g = g
        self.sindromes = []
        self.update_sindromes()

    def set_generator(self, g):
        self.g = g
        self.update_sindromes()

    def set_generator_lsb_first(self, g):
        self.g = np.flip(g)
        self.update_sindromes()

    def update_sindromes(self):
        # TODO generalize
        base = np.array([0 for i in range(len(self.g)-1)]+[1])
        self.sindromes.append(base)

    def decode(self, message):
        sindrome = self.pol_div(message, self.g)[1]
        changer = np.array([0 for i in range(len(self.g)-1)]+[1], dtype=int)
        rotations = 0

        # While sindrome is not all zeros
        while np.count_nonzero(sindrome) != 0:
            # Check if sindrome is in set of sindromes
            found = False
            for sind in self.sindromes:
                if np.array_equal(sindrome, sind):
                    message = np.mod(message+changer, 2)
                    sindrome = np.mod(sindrome + changer, 2)
                    found = True
            if found:
                continue

            # If sindrome not found in set
            message = self.rotate(message, 1)
            sindrome = self.rotate(sindrome, 1)
            rotations -= 1
            if sindrome[len(sindrome)-1] == 1:
                sindrome[len(sindrome) - 1] = 0
                sindrome = np.mod(sindrome+self.g, 2)
            print(sindrome, rotations)

        return self.rotate(self.pol_div(message, self.g)[0], rotations)

    @staticmethod
    def rotate(arr, rot):
        new_arr = [arr[(i+rot) % len(arr)] for i in range(len(arr))]
        return np.array(new_arr)

    @staticmethod
    def pol_div(n, d):
        def leading(p):
            for i in range(len(p)):
                if p[i] != 0:
                    return len(p)-i-1

        def shift(p, ind):
            aws = np.array([0 for i in range(len(p))], dtype=int)

            for i in range(len(p)):
                if (i-ind) >= 0:
                    aws[i-ind] = p[i]
            return aws

        len_dif = len(n) - len(d)

        assert len_dif >= 0

        d = np.concatenate([np.array([0 for i in range(len_dif)], dtype=int), d], 0)

        q = np.array([0 for i in range(len(n))], dtype=int)
        r = np.array(n)

        while np.count_nonzero(r) != 0 and leading(r) >= leading(d):
            t = shift(d, leading(r) - leading(d))
            q[len(q)-(leading(r) - leading(d))-1] = 1
            r = np.mod(r - t, 2)
        return q, r


if __name__ == "__main__":
    gen = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1])
    encoder = PolyEncoder(gen)
    decoder = PolyDecoder(gen)

    message = encoder.encode(np.array([1, 0, 1]))
    message = np.mod(message+np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]), 2)
    print(decoder.decode(message))
