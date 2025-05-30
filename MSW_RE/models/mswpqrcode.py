import random
import cv2
import torch
import numpy as np
from torch import Tensor
from numpy import ndarray
from models.msw import MSW
from typing import List, Tuple
from scipy.stats import pearsonr


class PEARSON_based_MSG(MSW):
    def __init__(self, N: int, q: int, delta: float, alpha: float):
        """
        Initializes the DCT-based MSG (Mixed-bit Sampling Watermarking) class.

        :param N: Size of each pattern (N x N).
        :param q: Base for encoding (e.g., 2 for binary, 8 for octal).
        :param delta: Watermark strength parameter.
        :param alpha: Target proportion of white pixels in the pattern.
        """
        super().__init__(N)
        self.q = q
        self.delta = delta
        self.alpha = alpha

    def generate_MSG(self, bits: list, index: int = 0) -> Tuple[ndarray, List[int]]:
        """
        Generates a watermark image based on the provided data.
        If the number of bits is not a perfect square, zeros are padded to form a complete matrix.

        :param bits: Input list of encoded bits.
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
        Computes the mixed-bit sampling watermark using dynamically generated patterns.
        Each block corresponds to one bit and uses its own pattern set.

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
                pattern_set = self.compute_pattern(gamma=self.delta, alpha=self.alpha, q=self.q, seed=seed)
                msg[h:h + self.N, w:w + self.N] = pattern_set[bit]
                index_bit += 1
        return np.uint8(msg * 255)

    def watermark_function(self, pattern_set: List[Tensor], gamma: float) -> Tensor:
        """
        Computes the watermark loss function based on inter-pattern correlation.
        This loss encourages the patterns to maintain a specified watermark strength.

        :param pattern_set: List of binary texture patterns (as tensors).
        :param gamma: Target watermark strength (Pearson correlation between different patterns).
        :return: Computed loss value.
        """
        loss = torch.tensor(0.)
        for i in range(len(pattern_set)):
            pattern_i = pattern_set[i]
            for j in range(i + 1, len(pattern_set)):
                pattern_j = pattern_set[j]
                mean_x, mean_y = torch.mean(pattern_i), torch.mean(pattern_j)
                cov_xy = torch.mean((pattern_i - mean_x) * (pattern_j - mean_y))
                std_x, std_y = torch.std(pattern_i, unbiased=False), torch.std(pattern_j, unbiased=False)
                loss += (cov_xy / (std_x * std_y + 1e-9) - gamma) ** 2
        loss /= len(pattern_set)
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

    def decode(self, ac_code: ndarray, pattern_list: list, v: int = 1) -> List[int]:
        """
        Decodes the embedded watermark from the input image block-by-block.
        Uses Pearson correlation to match each block with its most likely pattern.

        :param ac_code: Input watermarked image matrix (normalized to [0, 1]).
        :param pattern_list: List of predefined pattern sets for each bit position.
        :param v: Scale factor to resize patterns during matching.
        :return: Extracted bit sequence.
        """
        ext_bits = []
        index_bit = 0
        ac_code = ac_code / 255.0
        code_size, _ = ac_code.shape
        for h in range(0, code_size, int(self.N * v)):
            for w in range(0, code_size, int(self.N * v)):
                code_block = ac_code[h:h + int(self.N * v), w:w + int(self.N * v)]
                bit = self.extract(code_block, pattern_list[index_bit], v=v)
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
                pattern_set = self.compute_pattern(gamma=self.delta, alpha=self.alpha, q=self.q, seed=seed)
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

    def extract(self, ac_code_block: ndarray, pattern_set: ndarray, v: int = 1) -> int:
        """
        Extracts the embedded bit by finding the pattern with the highest Pearson correlation.

        :param v: Scale factor to enlarge each pattern (default is 1, meaning no scaling).
        :param ac_code_block: Block of coded data.
        :param pattern_set: Set of candidate patterns (shape: [num_patterns, H, W]).
        :return: Index of the best-matching pattern (interpreted as the extracted bit).
        """
        enlarged_patterns = []
        for pattern in pattern_set:
            enlarged = cv2.resize(pattern.numpy(), None, fx=v, fy=v, interpolation=cv2.INTER_NEAREST)
            enlarged_patterns.append(enlarged)
        bit = max(
            range(len(enlarged_patterns)),
            key=lambda i: pearsonr(enlarged_patterns[i].flatten(), ac_code_block.flatten())[0]
        )
        return bit



if __name__ == "__main__":
    q = 3
    data_len = 81
    data = [random.randint(0, q - 1) for _ in range(data_len)]
    pearson_cdp = PEARSON_based_MSG(3, q=q, alpha=0.583, delta=0.02)
    pearson_cdp.xi = 0.1
    pearson_cdp.theta = 0.01
    code, emb_bits = pearson_cdp.generate_MSG(data, index=10)
    pattern_list = pearson_cdp.compute_patterns(code, index=10)
    # code = cv2.resize(code, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    ext_bits = pearson_cdp.decode(code, pattern_list, v=1)
    corr = pearson_cdp.compute_score(code, pattern_list, v=1)
    print("Emb Bits:", emb_bits)
    print("Ext Bits:", ext_bits)
    print("Correlation Score:", corr)
