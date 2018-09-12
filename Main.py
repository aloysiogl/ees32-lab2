import matplotlib.pyplot as plt
import numpy as np
import time

from Channel import Channel
from DecoderHamming import DecoderHamming
from EncoderHamming import EncoderHamming
from Decoder import Decoder
from Encoder import Encoder

# Script which generates N random bits and simulates a random channel with probabilities ranging from 0.5 to 10e-6.
# It then plots a graph comparing different encoding processes.
# N = 240
N = 1000080

P6 = np.array([[1, 1, 1, 0, 0, 0],
               [1, 1, 0, 1, 0, 0],
               [1, 1, 0, 0, 1, 0],
               [1, 1, 0, 0, 0, 1],
               [1, 0, 1, 1, 0, 0],
               [1, 0, 1, 0, 1, 0],
               [1, 0, 1, 0, 0, 1],
               [1, 0, 0, 1, 1, 0]])

P9 = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0],
               [1, 1, 0, 0, 1, 1, 0, 0, 0],
               [1, 1, 0, 0, 0, 0, 1, 1, 0],
               [1, 0, 1, 0, 1, 0, 1, 0, 0],
               [1, 0, 1, 0, 0, 1, 0, 0, 1],
               [1, 0, 0, 1, 0, 1, 0, 1, 0],
               [1, 0, 0, 1, 0, 0, 1, 0, 1],
               [1, 0, 0, 0, 1, 0, 0, 1, 1],
               [0, 1, 1, 0, 0, 0, 0, 1, 1],
               [0, 1, 0, 1, 1, 0, 0, 0, 1],
               [0, 1, 0, 0, 0, 1, 1, 0, 1],
               [1, 1, 1, 1, 1, 1, 1, 1, 1]])

P12 = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                [1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
                [1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                [1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
                [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0],
                [1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                [1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
                [1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
                [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0],
                [1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1],
                [1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1]])

P15 = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                [1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                [1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                [1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0],
                [1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]])


def normal_process(codes, channels):
    outputs = [None] * len(channels)
    for c in range(len(channels)):
        outputs[c] = np.array([channels[c].add_noise(code) for code in codes])

    return outputs


def hamming_process(codes, channels):
    # Encoding
    hamming_encoder = EncoderHamming()
    encodes = [hamming_encoder.encode(code) for code in codes]

    # Channeling
    outputs = [None] * len(channels)
    for c in range(len(channels)):
        outputs[c] = np.array([channels[c].add_noise(code) for code in encodes])

    # Decoding
    hamming_decoder = DecoderHamming()
    for c in range(len(channels)):
        outputs[c] = np.array([hamming_decoder.decode(code) for code in outputs[c]])

    return outputs


def improved_process(P, codes, channels):
    improved_codes = []
    n = P.shape[1] // 3
    for c in range(len(codes) // n):
        improved_codes.append(np.concatenate(([codes[n * c + k] for k in range(n)])))

    # Encoding
    improved_encoder = Encoder(P)
    encodes = [improved_encoder.encode(code) for code in improved_codes]

    # Channeling
    outputs = [None] * len(channels)
    for c in range(len(channels)):
        outputs[c] = np.array([channels[c].add_noise(code) for code in encodes])

    # Decoding
    improved_decoder = Decoder(P, n + 1)
    for c in range(len(channels)):
        outputs[c] = np.array([improved_decoder.decode(code) for code in outputs[c]])

    return outputs


if __name__ == "__main__":
    t = time.time()

    # Generating random codes
    codes = []
    assert (N % 8 == 0 and N % 12 == 0 and N % 16 == 0 and N % 20 == 0)
    for i in range(0, N // 4):
        codes.append(np.rint(np.random.random_sample(4)).astype(bool))

    # Generating channels with different noises to plot a graph
    ps = [0.5, 0.2, 0.1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 2e-6]
    channels = [Channel(p) for p in ps]

    # Generating outputs without encoding, with hamming encoding and with our encoding
    normal_outputs = normal_process(codes, channels)
    hamming_outputs = hamming_process(codes, channels)
    improved_outputs6 = improved_process(P6, codes, channels)
    improved_outputs9 = improved_process(P9, codes, channels)
    improved_outputs12 = improved_process(P12, codes, channels)
    improved_outputs15 = improved_process(P15, codes, channels)

    # Comparing outputs and plotting a graph
    normal_ps = []
    hamming_ps = []
    improved_ps6 = []
    improved_ps9 = []
    improved_ps12 = []
    improved_ps15 = []
    for c in range(len(channels)):
        normal_ps.append(1 - np.count_nonzero(np.reshape(normal_outputs[c], (1, N)) == np.reshape(codes, (1, N))) / N)
        hamming_ps.append(1 - np.count_nonzero(np.reshape(hamming_outputs[c], (1, N)) == np.reshape(codes, (1, N))) / N)
        improved_ps6.append(
            1 - np.count_nonzero(np.reshape(improved_outputs6[c], (1, N)) == np.reshape(codes, (1, N))) / N)
        improved_ps9.append(
            1 - np.count_nonzero(np.reshape(improved_outputs9[c], (1, N)) == np.reshape(codes, (1, N))) / N)
        improved_ps12.append(
            1 - np.count_nonzero(np.reshape(improved_outputs12[c], (1, N)) == np.reshape(codes, (1, N))) / N)
        improved_ps15.append(
            1 - np.count_nonzero(np.reshape(improved_outputs15[c], (1, N)) == np.reshape(codes, (1, N))) / N)

    normal_ps = np.log(normal_ps) / np.log(10)
    hamming_ps = np.log(hamming_ps) / np.log(10)
    improved_ps6 = np.log(improved_ps6) / np.log(10)
    improved_ps9 = np.log(improved_ps9) / np.log(10)
    improved_ps12 = np.log(improved_ps12) / np.log(10)
    improved_ps15 = np.log(improved_ps15) / np.log(10)
    ps = np.log(ps) / np.log(10)

    print("Time taken:", time.time() - t, "s")
    fig, ax = plt.subplots()
    plt.xlim([0, -6])
    plt.xlabel("log(p)")
    plt.ylabel("log(Probabilidade de erro de bit)")
    plt1 = plt.plot(ps, normal_ps, label="Não codificado")
    plt2 = plt.plot(ps, hamming_ps, label="Hamming")
    plt3 = plt.plot(ps, improved_ps6, label="Código 14x6")
    plt4 = plt.plot(ps, improved_ps9, label="Código 21x9")
    plt5 = plt.plot(ps, improved_ps12, label="Código 28x12")
    plt6 = plt.plot(ps, improved_ps15, label="Código 35x15")
    ax.legend()
    plt.show()
