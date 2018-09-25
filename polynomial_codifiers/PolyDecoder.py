import numpy as np

class PolyDecoder:
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
