import torch
import hashlib
import numpy as np
from typing import List
from torch import Tensor
from numpy import ndarray
from abc import abstractmethod, ABC


class MSW(ABC):
    def __init__(self, N, xi: float = 0.1, theta: float = 0.001):
        """
        Initialize the MSW framework with given parameters.

        :param N: Dimension of the texture patterns (N x N).
        :param xi: Threshold for the ratio function.
        :param theta: Threshold for the watermark function.
        """
        self.xi = xi
        self.theta = theta
        self.N = N

    def bitstream_to_seed(self, bitstream):
        """
        Convert a bitstream into a seed value using SHA-256 hash.

        :param bitstream: Input bitstream.
        :return: Seed value for random number generation.
        """
        byte_stream = bytes(bitstream)
        sha256_hash = hashlib.sha256(byte_stream).digest()
        seed = int.from_bytes(sha256_hash[:4], byteorder='big', signed=False)
        return seed

    @abstractmethod
    def watermark_function(self, pattern_set, gamma):
        """
        Abstract method to be implemented for computing the watermark function.

        :param pattern_set: Set of texture patterns.
        :param gamma: Parameter for the watermark function.
        :return: Computed watermark value.
        """
        assert len(pattern_set) > 0

    @abstractmethod
    def ratio_function(self, pattern_set, alpha):
        """
        Abstract method to be implemented for computing the ratio function.

        :param pattern_set: Set of texture patterns.
        :param alpha: Proportion of white pixels.
        :return: Computed ratio value.
        """
        assert len(pattern_set) > 0

    def initilize_pattern(self, q, alpha):
        """
        Generate a set of binary texture patterns with a specified proportion of white pixels.

        :param q: Number of texture patterns to generate.
        :param alpha: Proportion of white pixels (value of 1) in each pattern.
        :return: A list of q binary texture patterns, each of size N x N.
        """
        pattern_set = []
        total_pixels = self.N * self.N
        num_white_pixels = int(round(total_pixels * alpha))
        num_black_pixels = total_pixels - num_white_pixels
        pixels = np.array([1] * num_white_pixels + [0] * num_black_pixels)
        for i in range(q):
            np.random.shuffle(pixels)
            pattern_i = pixels.reshape((self.N, self.N))
            pattern_set.append(torch.as_tensor(pattern_i, dtype=torch.float))
        return pattern_set

    def compute_gradients_for_ratio(self, pattern_set: List[Tensor], alpha: float) -> List[Tensor]:
        """
        Compute gradients for the ratio function.

        :param pattern_set: Set of texture patterns.
        :param alpha: Proportion of white pixels.
        :return: List of gradient tensors for each pattern.
        """
        grad_list = []
        ratio_loss = self.ratio_function(pattern_set, alpha)
        ratio_loss.backward()
        for pattern_tensor in pattern_set:
            grad_list.append(torch.sign(pattern_tensor.grad).detach())
        return grad_list

    def compute_gradients_for_watermark(self, pattern_set_t: List[ndarray], gamma: float):
        """
        Compute gradients for the watermark function.

        :param pattern_set_t: Set of texture patterns.
        :param gamma: Parameter for the watermark function.
        :return: List of gradient arrays for each pattern.
        """
        grad_list = []
        tensor_pattern_set_t = [torch.tensor(pattern, dtype=torch.float, requires_grad=True) for pattern in
                                pattern_set_t]
        watermark_loss = self.watermark_function(tensor_pattern_set_t, gamma)
        watermark_loss.backward()
        for pattern_tensor in tensor_pattern_set_t:
            grad_list.append(torch.sign(pattern_tensor.grad).detach().numpy())
        return grad_list

    def adjustable_coordinate_sets(self, pattern_i_t: ndarray, grad_matrix: ndarray):
        """
        Determine the coordinates of pixels that can be adjusted based on gradients.

        :param pattern_i_t: Input texture pattern.
        :param grad_matrix: Gradient matrix.
        :return: Arrays of x and y coordinates of adjustable pixels.
        """
        pattern_i_t = pattern_i_t.astype(int)
        ratio_P_1_one_xs, ratio_P_1_one_ys = np.where((pattern_i_t == 1) & (grad_matrix == +1))
        ratio_P_1_zero_xs, ratio_P_1_zero_ys = np.where((pattern_i_t == 0) & (grad_matrix == -1))
        xs = ratio_P_1_one_xs.tolist() + ratio_P_1_zero_xs.tolist()
        ys = ratio_P_1_one_ys.tolist() + ratio_P_1_zero_ys.tolist()
        return np.asarray(xs), np.asarray(ys)

    def uniformly_choose_corrdinate(self, xs, ys):
        """
        Uniformly choose a coordinate from the provided x and y arrays.

        :param xs: Array of x coordinates.
        :param ys: Array of y coordinates.
        :return: Chosen x and y coordinate.
        """
        x = None
        y = None
        if len(xs) > 0 and len(ys) > 0:
            indexes = np.random.randint(0, len(xs), 1)
            x = xs[indexes]
            y = ys[indexes]
        return x, y

    def adjust_black_white_ratio(self, pattern_i_t: ndarray, alpha: float):
        """
        Adjust the black-to-white pixel ratio of a texture pattern.

        :param pattern_i_t: Input texture pattern.
        :param alpha: Desired proportion of white pixels.
        :return: Adjusted texture pattern.
        """
        grad_mask = np.zeros_like(pattern_i_t, dtype=np.int32)
        condition = int(np.sum(pattern_i_t) > alpha * pattern_i_t.size)
        num_pixels_to_adjust = int(np.round(np.abs(np.sum(pattern_i_t) - alpha * pattern_i_t.size)))
        if num_pixels_to_adjust >= 1:
            xs, ys = np.where(pattern_i_t == condition)
            selected_indices = np.random.choice(range(len(xs)), size=num_pixels_to_adjust)
            grad_mask[xs[selected_indices], ys[selected_indices]] = (-1.) ** condition
        res_pattern_t = pattern_i_t + grad_mask
        return res_pattern_t

    def first_step(self, pattern_set_t: List[Tensor], alpha: float):
        """
        Perform the first adjustment step on the texture patterns.

        :param pattern_set_t: Set of texture patterns.
        :param alpha: Desired proportion of white pixels.
        :return: Adjusted set of texture patterns.
        """
        ratio_pattern_list = []
        for pattern_t in pattern_set_t:
            pattern_t_ = self.adjust_black_white_ratio(pattern_t.numpy(), alpha)
            ratio_pattern_list.append(pattern_t_)
        if len(ratio_pattern_list) == 0:
            ratio_pattern_list = [pattern.numpy() for pattern in pattern_set_t]
        return ratio_pattern_list

    def second_step(self, pattern_set_t: List[ndarray], gamma: float):
        """
        Perform the second adjustment step on the texture patterns.

        :param pattern_set_t: Set of texture patterns.
        :param gamma: Parameter for the watermark function.
        :return: Adjusted set of texture patterns.
        """
        result_list = []
        grad_w_list = self.compute_gradients_for_watermark(pattern_set_t, gamma)
        for pattern, grad in zip(pattern_set_t, grad_w_list):
            xs, ys = self.adjustable_coordinate_sets(pattern, grad)
            x, y = self.uniformly_choose_corrdinate(xs, ys)
            if x is not None and y is not None:
                pattern[x, y] = pattern[x, y] - grad[x, y]
                result_list.append(torch.as_tensor(pattern, dtype=torch.float))
            else:
                result_list.append(torch.as_tensor(pattern, dtype=torch.float))
        if len(result_list) == 0:
            result_list = [torch.tensor(pattern) for pattern in pattern_set_t]
        return result_list

    def compute_pattern(self, gamma: float, alpha: float, q: int, seed: int = 0):
        """
        Compute the final set of texture patterns that meet the specified ratio and watermark criteria.

        :param q: Number of texture patterns to generate.
        :param seed: Seed for random number generation.
        :param gamma: Parameter for the watermark function.
        :param alpha: Desired proportion of white pixels.
        :return: Final set of texture patterns.
        """
        assert q >= 2
        xi_, theta_ = 1e9, 1e9
        np.random.seed(seed)
        pattern_set_t = self.initilize_pattern(q, alpha)
        while xi_ > self.xi or theta_ > self.theta:
            pattern_set_t = self.first_step(pattern_set_t, alpha)
            pattern_set_t = self.second_step(pattern_set_t, gamma)
            xi_ = self.ratio_function(pattern_set_t, alpha).item()
            theta_ = self.watermark_function(pattern_set_t, gamma).item()
        return pattern_set_t
