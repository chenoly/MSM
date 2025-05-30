import random

import cv2
import torch
import numpy as np
from numpy import ndarray
from torch import Tensor
from typing import List, Tuple
from models.msw import MSW


class DCT_based_MSG(MSW):
    def __init__(self, N: int, alpha: float, gamma: float, coh_i: int = 1, coh_j: int = 2):
        """
        Initializes the DCT-based MSG (Mixed-bit Sampling Watermarking) class.

        :param N: Size of each pattern (N x N).
        :param alpha: Target proportion of white pixels in patterns.
        :param gamma: Watermark strength parameter.
        :param coh_i: DCT coefficient index for i-th direction.
        :param coh_j: DCT coefficient index for j-th direction.
        """
        super().__init__(N)
        self.q = 2
        self.K_x = None  # DCT basis matrix
        self.template_1 = None  # Template for bit '1'
        self.template_0 = None  # Template for bit '0'
        self.coh_i = coh_i  # Coefficient index i
        self.coh_j = coh_j  # Coefficient index j
        self.alpha = alpha  # Proportion of white pixels
        self.gamma = gamma  # Watermark strength
        self.InitParams()  # Initialize templates and DCT basis

    def InitParams(self):
        """
        Initializes the DCT basis matrix and binary templates used for watermark embedding.

        The templates are built using a specific DCT basis function. Pixels are set to 1 based on the sign of the basis value.
        """
        self.K_x = torch.zeros(size=(self.N, self.N), dtype=torch.float)
        self.template_1 = torch.zeros(size=(self.N, self.N), dtype=torch.float)
        self.template_0 = torch.zeros(size=(self.N, self.N), dtype=torch.float)

        for i in range(self.N):
            for j in range(self.N):
                b = (np.cos((2 * i + 1) * self.coh_i * np.pi / (2 * self.N)) *
                     np.cos((2 * j + 1) * self.coh_j * np.pi / (2 * self.N)) -
                     np.cos((2 * i + 1) * self.coh_j * np.pi / (2 * self.N)) *
                     np.cos((2 * j + 1) * self.coh_i * np.pi / (2 * self.N)))

                self.K_x[i, j] = b
                if b >= 0:
                    self.template_1[i, j] = 1
                else:
                    self.template_0[i, j] = 1

    def generate_MSG(self, bits: list, index: int = 0) -> Tuple[np.ndarray, List[int]]:
        """
        Generates a watermarked image by embedding the input bits into blocks using mixed sampling.

        If the number of bits is not a perfect square, zeros are padded to form a complete matrix.

        :param bits: Input data to be embedded as a watermark.
        :param index: Starting index for pattern generation.
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
        Computes the mixed-bit sampling watermark based on the embedded bits.

        Each block corresponds to one bit and uses its own pattern set derived from DCT templates.

        :param bits: List of bits to be embedded.
        :param index: Seed offset used in pattern generation.
        :return: Generated watermark image of size (L*N, L*N), where L = sqrt(len(bits)).
        """
        index_bit = 0
        L = int(np.sqrt(len(bits)))
        msg = np.zeros(shape=(self.N * L, self.N * L))
        for h in range(0, self.N * L, self.N):
            for w in range(0, self.N * L, self.N):
                bit = bits[index_bit]
                seed = int(np.cos((index + index_bit)) ** 2 * 1000)
                delta = (-1) ** (bit + 1) * self.gamma * (
                        bit * abs(np.sum(self.template_1.numpy() * self.K_x.numpy())) +
                        (1 - bit) * abs(np.sum(self.template_0.numpy() * self.K_x.numpy()))
                )
                pattern_set = self.compute_pattern(gamma=delta, alpha=self.alpha, q=2, seed=seed)
                pattern_i = self.template_1 * pattern_set[0] + self.template_0 * pattern_set[1]
                msg[h:h + self.N, w:w + self.N] = pattern_i
                index_bit += 1
        return np.uint8(msg * 255)

    def watermark_function(self, pattern_set: List[Tensor], delta: float) -> Tensor:
        """
        Computes the watermark loss function based on correlation with DCT basis.

        Encourages the pattern to maintain a specified watermark strength.

        :param pattern_set: List of binary texture patterns (as tensors).
        :param delta: Target DCT projection value for watermark strength.
        :return: Computed loss value.
        """
        pattern = self.template_1 * pattern_set[0] + self.template_0 * pattern_set[1]
        delta_t = torch.sum(pattern * self.K_x)
        loss = (delta_t - delta) ** 2
        return loss

    def ratio_function(self, pattern_set: List[Tensor], alpha: float) -> Tensor:
        """
        Computes the ratio loss function that enforces a target pixel intensity distribution.

        Encourages each pattern to have a specific proportion of white pixels.

        :param pattern_set: List of binary texture patterns (as tensors).
        :param alpha: Target proportion of white pixels.
        :return: Computed loss value.
        """
        loss = torch.tensor(0.)
        for pattern in pattern_set:
            alpha_t = pattern.sum() / pattern.numel()
            loss += (alpha_t - alpha) ** 2
        loss /= len(pattern_set)
        return loss

    def decode(self, ac_code: np.ndarray) -> List[int]:
        """
        Decodes the embedded watermark from the input image block-by-block.

        Uses Pearson correlation to match each block with its most likely pattern.

        :param ac_code: Input watermarked image matrix (normalized to [0, 1]).
        :param index: Seed offset used during decoding.
        :return: A tuple containing:
                 - Extracted bit sequence,
                 - Average authentication score across all blocks.
        """
        ext_bits = []
        index_bit = 0
        ac_code = ac_code / 255.0
        code_size, _ = ac_code.shape
        for h in range(0, code_size, self.N):
            for w in range(0, code_size, self.N):
                code_block = ac_code[h:h + self.N, w:w + self.N]
                bit = self.Extract(code_block)
                ext_bits.append(bit)
                index_bit += 1
        return ext_bits

    def compute_patterns(self, ac_code: ndarray, index: int = 0) -> List[List[np.ndarray]]:
        """
        Pre-computes pattern sets for each block in the image for decoding purposes.
        Pattern sets are generated using a deterministic seed derived from the index.

        :param ac_code: Input watermarked image matrix (normalized to [0, 1]).
        :param index: Seed offset used during decoding.
        :return: List of pattern sets for each block.
        """
        index_bit = 0
        pattern_list = []
        ac_code = ac_code / 255.0
        code_size, _ = ac_code.shape
        for h in range(0, code_size, self.N):
            for w in range(0, code_size, self.N):
                seed = int(np.cos((index + index_bit)) ** 2 * 1000)
                bit = 1
                delta = (-1) ** (bit + 1) * self.gamma * (
                        bit * abs(np.sum(self.template_1.numpy() * self.K_x.numpy())) +
                        (1 - bit) * abs(np.sum(self.template_0.numpy() * self.K_x.numpy()))
                )
                pattern_1_set = self.compute_pattern(gamma=delta, alpha=self.alpha, q=2, seed=seed)
                pattern_1_i = self.template_1 * pattern_1_set[0] + self.template_0 * pattern_1_set[1]

                seed = int(np.cos((index + index_bit)) ** 2 * 1000)
                bit = 0
                delta = (-1) ** (bit + 1) * self.gamma * (
                        bit * abs(np.sum(self.template_1.numpy() * self.K_x.numpy())) +
                        (1 - bit) * abs(np.sum(self.template_0.numpy() * self.K_x.numpy()))
                )
                pattern_0_set = self.compute_pattern(gamma=delta, alpha=self.alpha, q=2, seed=seed)
                pattern_0_i = self.template_1 * pattern_0_set[0] + self.template_0 * pattern_0_set[1]
                pattern_set = [pattern_1_i, pattern_0_i]
                pattern_list.append(pattern_set)
                index_bit += 1
        return pattern_list

    def compute_score(self, ac_code: ndarray, pattern_list: list, v: int = 1) -> float:
        """
        Computes an overall distinguishability score for the watermarked image.
        This score is based on how well each block matches its expected pattern.

        :param ac_code: Input watermarked image matrix (normalized to [0, 1]).
        :param pattern_list: List of predefined pattern sets for each block.
        :param v: Scale factor to resize patterns during matching.
        :return: Mean distinguishability score across all blocks.
        """
        scores = []
        index_patterns = 0
        ac_code = ac_code / 255.0
        code_size, _ = ac_code.shape
        for h in range(0, code_size, int(self.N * v)):
            for w in range(0, code_size, int(self.N * v)):
                code_block = ac_code[h:h + int(self.N * v), w:w + int(self.N * v)]
                score = self._2lqrcode_score(code_block.flatten(), pattern_list[index_patterns], v=v)
                index_patterns += 1
                scores.append(score)
        mean_score = sum(scores) / len(scores)
        return mean_score

    def _2lqrcode_score(self, current: np.ndarray, digital_patterns: list, v: int = 1) -> float:
        """
        Calculates the minimum difference between the highest Pearson correlation and other correlations
        between the current block and a set of predefined digital patterns.
        This score evaluates how distinguishable the best-matched pattern is from others,
        serving as a robustness metric for authentication.

        :param current: Flattened array representing the current image block.
        :param digital_patterns: List of pattern templates (as flattened arrays).
        :param v: Scale factor to enlarge each pattern.
        :return: Minimum absolute difference between max correlation and others.
        """
        max_corr_values = []  # Store Pearson correlation values for each template
        for template in digital_patterns:
            template = cv2.resize(template.numpy(), None, fx=v, fy=v, interpolation=cv2.INTER_NEAREST)
            corr = np.corrcoef(current, template.flatten())[0, 1]  # Get correlation coefficient
            max_corr_values.append(corr)
        max_corr = max(max_corr_values)
        max_corr_values.remove(max_corr)
        min_diff = min(abs(max_corr - c) for c in max_corr_values)
        return min_diff

    def Extract(self, ac_code_block: np.ndarray) -> int:
        """
        Extracts the embedded bit by computing the inner product with the DCT basis.

        Positive result means bit '1', negative means bit '0'.

        :param ac_code_block: Block of coded data.
        :return: Extracted bit (0 or 1).
        """
        delta_c = np.sum(ac_code_block * self.K_x.numpy())
        return 1 if delta_c >= 0 else 0


if __name__ == "__main__":
    data_len = 81
    q = 2
    data = [random.randint(0, q - 1) for _ in range(data_len)]
    dct_cdp = DCT_based_MSG(12, alpha=0.583, gamma=0.01)
    code, bits_embed = dct_cdp.generate_MSG(data, index=1)
    pattern_list = dct_cdp.compute_patterns(code, index=1)
    score = dct_cdp.compute_score(code, pattern_list, v=1)
    ext_data = dct_cdp.decode(code)
    print(data)
    print(bits_embed)
    print(ext_data)
    print(score)
