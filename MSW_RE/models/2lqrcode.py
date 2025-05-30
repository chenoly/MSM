from typing import Tuple

import numpy as np
from numpy import ndarray, uint8

p1 = np.array([[1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
               [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
               [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
               [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
               [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
               [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
               [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
               [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0],
               [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
               [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
               [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]])
p2 = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
               [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
               [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
               [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0],
               [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
               [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
               [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
               [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
               [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0],
               [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
               [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]])
p3 = np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
               [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
               [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
               [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
               [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
               [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
               [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
               [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
               [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
               [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
               [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
               [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]])


class TwoLQRcode:
    def __init__(self):
        """
        Initialize the QR code generator with three predefined patterns (p1, p2, p3).
        Each pattern is a 12x12 binary-like matrix.
        """
        self.patterns = [p1, p2, p3]  # 0 -> p1, 1 -> p2, 2 -> p3
        self.block_size = 12  # Size of each pattern block
        self.q = 3

    def generate_MSG(self, data: list, index: int = 0) -> Tuple[uint8, None]:
        """
        Generate a large QR-like matrix by tiling selected patterns based on input data.

        :param index: the index for generating image
        :param data: List of integers (0, 1 or 2), where each number corresponds to a pattern.
        :return: A 2D NumPy array representing the generated matrix.
        """
        num_blocks = int(np.sqrt(len(data)))
        if num_blocks ** 2 != len(data):
            raise ValueError("data length must be a perfect square")

        big_matrix = []
        for row in range(num_blocks):
            matrix_row = []
            start_index = row * num_blocks
            end_index = start_index + num_blocks
            row_data = data[start_index:end_index]
            for val in row_data:
                matrix_row.append(self.patterns[val])
            big_matrix.append(np.hstack(matrix_row))
        msg = np.vstack(big_matrix)
        return np.uint8(msg * 255), None

    def decode(self, scanned_code: ndarray) -> (list, float):
        """
        Decode a scanned QR-like matrix into the original data sequence,
        using Pearson Correlation Coefficient to match each block to the closest pattern.

        :param scanned_code: A 2D NumPy array (N x N) representing the scanned matrix.
        :return: A list of integers (0, 1 or 2) indicating which pattern was used for each block.
        """
        height, width = scanned_code.shape
        if height != width:
            raise ValueError("scanned_code must be a square matrix")
        if height % self.block_size != 0:
            raise ValueError(f"Matrix size must be a multiple of {self.block_size}")
        num_blocks = height // self.block_size
        decoded_data = []
        score_list = []
        for row in range(num_blocks):
            for col in range(num_blocks):
                # Extract current block
                start_row = row * self.block_size
                end_row = start_row + self.block_size
                start_col = col * self.block_size
                end_col = start_col + self.block_size
                block = scanned_code[start_row:end_row, start_col:end_col]
                min_diff = self._2lqrcode_score(block.flatten())
                score_list.append(min_diff)
                # Flatten the block for correlation calculation
                flat_block = block.flatten()
                best_match = None
                best_corr = -1.0  # Minimum possible value for Pearson correlation
                for i, pattern in enumerate(self.patterns):
                    flat_pattern = pattern.flatten()
                    # Compute Pearson Correlation Coefficient
                    corr = np.corrcoef(flat_block, flat_pattern)[0, 1]
                    if corr > best_corr:
                        best_corr = corr
                        best_match = i
                decoded_data.append(best_match)
        mean_score = sum(score_list) / len(score_list)
        return decoded_data, mean_score

    def _2lqrcode_score(self, current):
        max_corr_values = []  # Store max corr values for this block
        for template in self.patterns:
            flat_template = template.flatten()
            corr = np.corrcoef(current, flat_template)[0, 1]
            max_corr_values.append(corr)
        max_corr = max(max_corr_values)
        max_corr_values.remove(max_corr)
        min_diff = min(abs(max_corr - c) for c in max_corr_values)
        return min_diff


if __name__ == "__main__":
    qr_generator = TwoLQRcode()
    data = [0, 1, 2, 0]
    generated_code = qr_generator.generate_MSG(data)
    decoded_data, score = qr_generator.decode(generated_code)
    print(decoded_data)
    print(f"Authentication Score (p_bar): {score:.4f}")
