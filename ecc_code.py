import random
import string
import bchlib


class BCH:
    def __init__(self, BCH_POLYNOMIAL_=487, BCH_BITS_=5):
        """
        Initialize the BCH encoder/decoder with specified polynomial and bit size.

        :param BCH_POLYNOMIAL_: Polynomial used for BCH encoding.
        :param BCH_BITS_: Bit size used for BCH encoding.
        """
        self.bch = bchlib.BCH(BCH_POLYNOMIAL_, BCH_BITS_)

    def Encode(self, data_: bytearray):
        """
        Encode the given data using BCH encoding and generate a binary secret.

        :param data_: Data to be encoded as a bytearray.
        :return: Encoded data as a list of binary digits (0 and 1).
        """
        ecc = self.bch.encode(data_)  # Encode data and generate error-correcting code (ECC)
        packet = data_ + ecc  # Append ECC to the original data
        packet_binary = ''.join(format(x, '08b') for x in packet)  # Convert to binary string
        secret_ = [int(x) for x in packet_binary]  # Convert binary string to a list of integers (0 and 1)
        return secret_

    def Decode(self, secret_: list):
        """
        Decode the given binary secret using BCH decoding.

        :param secret_: Encoded data as a list of binary digits (0 and 1).
        :return: Decoded data as a bytearray if successful, None otherwise.
        """
        # Convert list of binary digits to binary string
        packet_binary = "".join([str(int(bit)) for bit in secret_])
        # Convert binary string to bytearray
        packet = bytes(int(packet_binary[i: i + 8], 2) for i in range(0, len(packet_binary), 8))
        packet = bytearray(packet)
        data_, ecc = packet[:-self.bch.ecc_bytes], packet[-self.bch.ecc_bytes:]  # Split data and ECC
        bit_flips = self.bch.decode(data_, ecc)  # Decode and correct errors
        if bit_flips[0] != -1:
            return bit_flips[1]  # Return decoded data if successful
        return None  # Return None if decoding failed


def generate_random_string(n):
    """
    Generate a random string of specified length.

    :param n: Length of the random string.
    :return: Randomly generated string containing letters and digits.
    """
    characters = string.ascii_letters + string.digits  # Characters set: letters and digits
    random_string = ''.join(random.choice(characters) for _ in range(n))  # Generate random string
    return random_string
