import numpy as np
from polynomial_codifiers.PolyEncoder import PolyEncoder


class PolyDecoder:
    def __init__(self, g, codeword_length):
        self.g = g
        self.sindromes = []
        self.update_sindromes(codeword_length)

    def set_generator(self, g):
        self.g = g
        self.update_sindromes()

    def set_generator_lsb_first(self, g):
        self.g = np.flip(g)
        self.update_sindromes()

    def update_sindromes(self, codeword_length):
        # TODO generalize
        base = self.calc_sindrome(np.array([0 for i in range(codeword_length - 1)]+[1]))
        print('base', base)
        self.sindromes.append(base)

    def calc_sindrome(self, message):
        sindrome = self.reduce_degree(self.pol_div(message, self.g)[1], self.degree(self.g))
        print('calculated', sindrome)
        return sindrome

    def decode(self, message):
        g = self.reduce_degree(self.g, self.degree(self.g))

        sindrome = self.calc_sindrome(message)
        print('initial sindrome', sindrome)
        # print(message)
        # print(self.g)
        # print(self.pol_div(message, self.g))
        print("sindome")
        print(sindrome)
        print("g")
        print(g)
        print("------")
        # exit()

        changer = np.array([0 for i in range(len(g)-1)]+[1], dtype=int)
        message_changer = np.array([0 for i in range(len(message)-1)]+[1], dtype=int)
        rotations = 0

        # While sindrome is not all zeros
        while np.count_nonzero(sindrome) != 0:
            # Check if sindrome is in set of sindromes
            found = False
            for sind in self.sindromes:
                if np.array_equal(sindrome, sind):
                    message = np.mod(message+message_changer, 2)
                    # sindrome = np.mod(sindrome + changer, 2)
                    print('pre-found', sindrome)
                    sindrome = self.calc_sindrome(message)
                    print('found', sindrome)
                    found = True
                    break
            if found:
                continue

            # If sindrome not found in set
            message = self.rotate(message, -1)
            sindrome = self.rotate(sindrome, -1)
            rotations += 1
            if sindrome[0] == 1:
                sindrome = np.mod(sindrome+g, 2)
                sindrome[0] = 0

        return self.pol_div(self.rotate(message, rotations), self.g)[0]

    @staticmethod
    def degree(p):
        for i in range(len(p)):
            if p[i] != 0:
                return len(p) - i

    @staticmethod
    def reduce_degree(p, d):
        arr = [0 for i in range(d)]
        for i in range(d):
            arr[d-i-1] = p[len(p)-i-1]
        return np.array(arr)

    @staticmethod
    def rotate(arr, rot):
        new_arr = [arr[(i-rot) % len(arr)] for i in range(len(arr))]
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
        
        partial = np.polymul(d, q)
        assert np.array_equal(n, np.mod(np.concatenate([np.array([0 for i in range(len(n) - len(partial))]), partial], 0)+r, 2))
        return q, r


if __name__ == "__main__":
    # n=14
    # g=1+x^2+x^6 (divisor de X^14 + 1 : OK)
    gen = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1])
    encoder = PolyEncoder(gen)
    decoder = PolyDecoder(gen, codeword_length=14)

    message = encoder.encode(np.array([1, 0, 1]))

    message = np.mod(message+np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 2)
    print('codeword', message)
    print('begin decoding')
    decoded = decoder.decode(message)
    print('decoded', decoded)

    # from preprocessing.MatrixReader import MatrixReader
    #
    # reader = MatrixReader()
    # reader.read()
    # print(np.dot(np.transpose(np.array([
    #     [1],
    #     [0],
    #     [1],
    #     [0],
    #     [0],
    #     [0],
    #     [0],
    #     [0],
    # ])), reader.get_matrix(5)))
