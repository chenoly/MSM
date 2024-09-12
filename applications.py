import math

import cv2
import torch
import numpy as np
from model import MSW
from typing import List
from torch import Tensor
from ecc_code import BCH
from numpy import ndarray
from scipy.stats import pearsonr
from Crypto.Util.number import bytes_to_long


class DCT_based_MSG(MSW):
    def __init__(self, N: int, coh_i: int = 1, coh_j: int = 2):
        """
        Initializes the DCT-based MSG (Mixed-bit Sampling Watermarking) class.

        :param N: Size of the patterns (N x N).
        :param coh_i: Coefficient for the i-th direction in DCT.
        :param coh_j: Coefficient for the j-th direction in DCT.
        """
        super().__init__(N)
        self.K_x = None
        self.bch = BCH()
        self.template_1 = None
        self.template_0 = None
        self.coh_i = coh_i
        self.coh_j = coh_j
        self.InitParams()

    def InitParams(self):
        """
        Initializes the DCT matrix and templates used for watermarking.
        """
        self.K_x = torch.zeros(size=(self.N, self.N), dtype=torch.float)
        self.template_1 = torch.zeros(size=(self.N, self.N), dtype=torch.float)
        self.template_0 = torch.zeros(size=(self.N, self.N), dtype=torch.float)

        for i in range(self.N):
            for j in range(self.N):
                b = (np.cos((2 * i + 1) * self.coh_i * np.pi / (2 * self.N)) * np.cos(
                    (2 * j + 1) * self.coh_j * np.pi / (2 * self.N)) -
                     np.cos((2 * i + 1) * self.coh_j * np.pi / (2 * self.N)) * np.cos(
                            (2 * j + 1) * self.coh_i * np.pi / (2 * self.N)))
                self.K_x[i, j] = b
                if b >= 0:
                    self.template_1[i, j] = 1
                else:
                    self.template_0[i, j] = 1

    def generate_MSG(self, data: str, alpha: float = 0.7, gamma: float = 0.06, index: int = 0) -> (
            np.ndarray, List[int]):
        """
        Generates a watermark image based on the provided data.

        :param data: Input data to be embedded as a watermark.
        :param alpha: Proportion of white pixels in the pattern.
        :param gamma: Watermark strength parameter.
        :param index: Starting index for pattern generation.
        :return: Tuple containing the generated watermark image and the embedded bits.
        """
        byte_array = bytearray(data.encode('utf-8'))
        bits = self.bch.Encode(byte_array)
        N = round(np.sqrt(len(bits)))
        com_N = N ** 2 - len(bits)
        bits_embed = bits + [0 for _ in range(com_N)]
        msg = self.compute_msw(bits_embed, alpha, gamma, index)
        return msg, bits_embed

    def compute_msw(self, bits: List[int], alpha: float, gamma: float, index: int) -> np.ndarray:
        """
        Computes the mixed-bit sampling watermark based on the embedded bits.

        :param bits: List of bits to be embedded.
        :param alpha: Proportion of white pixels in the pattern.
        :param gamma: Watermark strength parameter.
        :param index: Starting index for pattern generation.
        :return: Watermarked image as a numpy array.
        """
        index_bit = 0
        L = int(np.sqrt(len(bits)))
        msg = np.zeros(shape=(self.N * L, self.N * L))
        for h in range(0, self.N * L, self.N):
            for w in range(0, self.N * L, self.N):
                bit = bits[index_bit]
                seed = index + index_bit
                delta = (-1) ** (bit + 1) * gamma * (
                        bit * abs(np.sum(self.template_1.numpy() * self.K_x.numpy())) +
                        (1 - bit) * abs(np.sum(self.template_0.numpy() * self.K_x.numpy()))
                )
                pattern_set = self.compute_pattern(gamma=delta, alpha=alpha, q=2, seed=seed)
                pattern_i = self.template_1 * pattern_set[0] + self.template_0 * pattern_set[1]
                msg[h:h + self.N, w:w + self.N] = pattern_i
                index_bit += 1
        return np.uint8(msg * 255)

    def watermark_function(self, pattern_set: List[Tensor], delta: float) -> Tensor:
        """
        Computes the watermark loss function.

        :param pattern_set: List of binary texture patterns.
        :param delta: Target watermark strength.
        :return: Computed loss value.
        """
        pattern = self.template_1 * pattern_set[0] + self.template_0 * pattern_set[1]
        delta_t = torch.sum(pattern * self.K_x)
        loss = (delta_t - delta) ** 2
        return loss

    def ratio_function(self, pattern_set: List[Tensor], alpha: float) -> Tensor:
        """
        Computes the ratio loss function.

        :param pattern_set: List of binary texture patterns.
        :param alpha: Target proportion of white pixels.
        :return: Computed loss value.
        """
        loss = torch.tensor(0.)
        for pattern in pattern_set:
            alpha_t = pattern.sum() / pattern.numel()
            loss += (alpha_t - alpha) ** 2
        loss /= len(pattern_set)
        return loss

    def decode(self, ac_code: np.ndarray) -> (str, List[int]):
        """
        Decodes the watermark from the coded data.

        :param ac_code: Input array of coded data (normalized to [0, 1]).
        :return: Tuple containing the extracted data and the list of extracted bits.
        """
        ext_bits = []
        ac_code = ac_code / 255.  # Normalize to [0, 1]
        code_size, _ = ac_code.shape
        for h in range(0, code_size, self.N):
            for w in range(0, code_size, self.N):
                code_block = ac_code[h:h + self.N, w:w + self.N]
                bit = self.Extract(code_block)
                ext_bits.append(bit)
        end_bit = len(ext_bits) // 8 * 8
        ext_data = self.bch.Decode(ext_bits[0:end_bit])
        return ext_data.decode('utf-8'), ext_bits

    def Extract(self, ac_code_block: np.ndarray) -> int:
        """
        Extracts a single bit from a code block.

        :param ac_code_block: Block of coded data.
        :return: Extracted bit (0 or 1).
        """
        delta_c = np.sum(ac_code_block * self.K_x.numpy())
        return 1 if delta_c >= 0 else 0


# dct_cdp = DCT_based_MSG(36)
# data = "123"
# code, bits_embed = dct_cdp.generate_MSG(data)
# ext_data, ext_bits = dct_cdp.decode(code)
# print(data, bits_embed)
# print(ext_data, ext_bits)


class PEARSON_based_MSG(MSW):
    def __init__(self, N: int):
        """
        Initializes the DCT-based MSG (Mixed-bit Sampling Watermarking) class.

        :param N: Size of the patterns (N x N).
        :param coh_i: Coefficient for the i-th direction in DCT.
        :param coh_j: Coefficient for the j-th direction in DCT.
        """
        super().__init__(N)
        self.bch = BCH()

    def string_to_base_q_bits(self, data: str, q: int) -> list:
        """
        Converts a string to a list of q-base bits.

        :param data: Input string to be converted.
        :param q: Base for the encoding (e.g., 2 for binary, 8 for octal).
        :return: List of q-base bits representing the input string.
        """
        # Convert string to bytes
        byte_data = data.encode('utf-8')

        # Convert bytes to a long integer
        integer_value = bytes_to_long(byte_data)

        # Calculate the number of digits needed in base q
        num_digits = math.ceil(math.log(integer_value + 1, q))

        # Convert integer to base-q representation
        base_q_bits = []
        while integer_value > 0:
            base_q_bits.append(integer_value % q)
            integer_value //= q

        # Pad the list to ensure it's of the desired length
        while len(base_q_bits) < num_digits:
            base_q_bits.append(0)

        return base_q_bits[::-1]  # Reverse the list to match the order of digits

    def base_q_bits_to_string(self, base_q_bits: list, q: int) -> str:
        """
        Converts a list of q-base bits to a string, removing any excess bits.

        :param base_q_bits: List of q-base bits to be decoded.
        :param q: Base for the encoding (e.g., 2 for binary, 3 for ternary).
        :return: Decoded string.
        """

        # Determine the number of digits needed for the base-q representation
        num_digits = math.ceil(len(base_q_bits) / math.log(q, 2))
        # Truncate base_q_bits to fit into num_digits base-q digits
        truncated_bits = base_q_bits[:num_digits]

        # Convert truncated base-q bits to an integer
        integer_value = 0
        for bit in truncated_bits:
            integer_value = integer_value * q + bit

        # Convert integer to byte array
        byte_length = (integer_value.bit_length() + 7) // 8
        byte_data = integer_value.to_bytes(byte_length, byteorder='big')

        # Convert byte data to string
        try:
            decoded_string = byte_data.decode('utf-8')
        except UnicodeDecodeError:
            decoded_string = byte_data.decode('utf-8', errors='replace')

        return decoded_string

    def generate_MSG(self, data: str, alpha: float = 0.5, delta: float = 0.0, q: int = 3, index: int = 0) -> (
            np.ndarray, List[int]):
        """
        Generates a watermark image based on the provided data.

        :param q:
        :param data: Input data to be embedded as a watermark.
        :param alpha: Proportion of white pixels in the pattern.
        :param delta: Watermark strength parameter.
        :param index: Starting index for pattern generation.
        :return: Tuple containing the generated watermark image and the embedded bits.
        """
        bits = self.string_to_base_q_bits(data, q)
        N = round(np.sqrt(len(bits)))
        com_N = N ** 2 - len(bits)
        bits_embed = bits + [0 for _ in range(com_N)]
        msg = self.compute_msw(bits_embed, alpha, delta, q, index)
        return msg, bits_embed

    def compute_msw(self, bits: List[int], alpha: float, gamma: float, q: int, index: int) -> np.ndarray:
        """
        Computes the mixed-bit sampling watermark based on the embedded bits.

        :param q:
        :param bits: List of bits to be embedded.
        :param alpha: Proportion of white pixels in the pattern.
        :param gamma: Watermark strength parameter.
        :param index: Starting index for pattern generation.
        :return: Watermarked image as a numpy array.
        """
        index_bit = 0
        L = int(np.sqrt(len(bits)))
        msg = np.zeros(shape=(self.N * L, self.N * L))
        for h in range(0, self.N * L, self.N):
            for w in range(0, self.N * L, self.N):
                bit = bits[index_bit]
                seed = index + index_bit
                pattern_set = self.compute_pattern(gamma=gamma, alpha=alpha, q=q, seed=seed)
                msg[h:h + self.N, w:w + self.N] = pattern_set[bit]
                index_bit += 1
        return np.uint8(msg * 255)

    def watermark_function(self, pattern_set: List[Tensor], gamma: float) -> Tensor:
        """
        Computes the watermark loss function.

        :param pattern_set: List of binary texture patterns.
        :param gamma: Target watermark strength.
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
        Computes the ratio loss function.

        :param pattern_set: List of binary texture patterns.
        :param alpha: Target proportion of white pixels.
        :return: Computed loss value.
        """
        loss = torch.tensor(0.)
        for pattern in pattern_set:
            alpha_t = pattern.sum() / pattern.numel()
            loss += (alpha_t - alpha) ** 2
        loss /= len(pattern_set)
        return loss

    def decode(self, ac_code: np.ndarray, alpha: float = 0.5, delta: float = 0.0, q: int = 3, index: int = 0) -> (
            str, List[int]):
        """
        Decodes the watermark from the coded data.

        :param alpha:
        :param delta:
        :param q:
        :param index:
        :param ac_code: Input array of coded data (normalized to [0, 1]).
        :return: Tuple containing the extracted data and the list of extracted bits.
        """
        ext_bits = []
        index_bit = 0
        ac_code = ac_code / 255.  # Normalize to [0, 1]
        code_size, _ = ac_code.shape
        for h in range(0, code_size, self.N):
            for w in range(0, code_size, self.N):
                code_block = ac_code[h:h + self.N, w:w + self.N]
                seed = index + index_bit
                pattern_set = self.compute_pattern(gamma=delta, alpha=alpha, q=q, seed=seed)
                bit = self.Extract(code_block, pattern_set)
                ext_bits.append(bit)
                index_bit += 1
        ext_data = self.base_q_bits_to_string(ext_bits, q)
        return ext_data, ext_bits

    def Extract(self, ac_code_block: np.ndarray, pattern_set: ndarray) -> int:
        """
        Extracts a single bit from a code block.

        :param pattern_set:
        :param ac_code_block: Block of coded data.
        :return: Extracted bit (0 or 1).
        """
        bit = max(range(len(pattern_set)), key=lambda i: pearsonr(pattern_set[i].flatten(), ac_code_block.flatten())[0])
        return bit


# pearson_cdp = PEARSON_based_MSG(36)
# data = "123"
# q = 3
# code, bits_embed = pearson_cdp.generate_MSG(data, alpha=0.7, gamma=0.0, q=q)
# ext_data, ext_bits = pearson_cdp.decode(code, alpha=0.7, gamma=0.0, q=q)
# print(data, bits_embed)
# print(ext_data, ext_bits)
