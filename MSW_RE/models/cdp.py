import numpy as np
from typing import List, Tuple


class CDP:
    def __init__(self, N: int, alpha: float, gamma: float):
        """
        Initializes the CDP (Copy-Detection Pattern) watermarking class.

        This class generates binary watermarked images with a specified pattern density and watermark strength.

        :param N: Size of each watermark block (N x N).
        :param alpha: Target proportion of white pixels in patterns.
        :param gamma: Watermark strength parameter (not used currently, reserved for future extensions).
        """
        self.N = N  # Block size
        self.q = 2  # Binary pattern (0 or 1)
        self.alpha = alpha  # Proportion of white pixels
        self.gamma = gamma  # Watermark strength (currently not used)

    def generate_MSG(self, bits: list, index: int = 0) -> Tuple[np.ndarray, List[int]]:
        """
        Generates a watermarked image by embedding the input bit sequence.

        If the number of bits is not a perfect square, zeros are padded to form a complete matrix.

        :param bits: Input data to be embedded as a watermark.
        :param index: Optional seed offset for pattern generation (reserved for future use).
        :return: A tuple containing:
                 - The generated watermark image (as uint8 array),
                 - The embedded bit sequence (including padding).
        """
        N = round(np.sqrt(len(bits)))
        com_N = N ** 2 - len(bits)
        bits_embed = bits + [0 for _ in range(com_N)]
        msg = self.compute_msw(bits_embed, index)
        return msg, bits_embed

    def compute_msw(self, bits: List[int], index: int) -> np.ndarray:
        """
        Computes the mixed-bit sampling watermark using random binary patterns.

        Each block corresponds to one bit. Currently uses binomial noise as dummy watermark content.

        :param bits: List of bits to be embedded.
        :param index: Seed offset used in pattern generation (not used here yet).
        :return: Generated watermark image of size (L*N, L*N), where L = sqrt(len(bits)).
        """
        index_bit = 0
        L = int(np.sqrt(len(bits)))
        msg = np.zeros(shape=(self.N * L, self.N * L))

        for h in range(0, self.N * L, self.N):
            for w in range(0, self.N * L, self.N):
                # Generate a random binary block based on alpha (white pixel ratio)
                marked_code = np.random.binomial(1, p=self.alpha, size=(self.N, self.N))
                msg[h:h + self.N, w:w + self.N] = marked_code
                index_bit += 1

        return np.uint8(msg * 255)  # Convert to 8-bit grayscale format

    def decode(self, ac_code: np.ndarray, index: int = 0):
        """
        Decodes the embedded watermark from the coded image (currently unimplemented).

        Reserved for future implementation of decoding logic.

        :param ac_code: Input watermarked image matrix (normalized to [0, 1]).
        :param index: Seed offset used during decoding (optional).
        :return: None or placeholder; will be updated later.
        """
        pass
