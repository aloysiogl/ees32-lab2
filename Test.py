import numpy as np
from EncoderHamming import EncoderHamming
from DecoderHamming import DecoderHamming
from Encoder import Encoder
from Decoder import Decoder
from Channel import Channel

if __name__ == '__main__':
    mode = 0

    # Instantiating encoder, decoder and channel
    if mode == 0:
        P = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0],
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
        encoder = Encoder(P)
        decoder = Decoder(P, 3)
    else:
        encoder = EncoderHamming()
        decoder = DecoderHamming()
    channel = Channel(0.3)

    # Receiving code
    code = np.matrix([[int(x) for x in input()]])

    # Printing code
    print("Code received: ")
    print(code)
    input()

    # Encoding channel
    print("Encoding: ")
    encoded = encoder.encode(code)
    print(encoded)
    input()

    # Passing through channel
    print("Passing through channel: ")
    through_channel = np.array(channel.add_noise(np.array(encoded)[0]))
    print(through_channel)
    input()

    # Decoding code
    print("Decoding: ")
    decoded = decoder.decode(through_channel)
    print(decoded)
