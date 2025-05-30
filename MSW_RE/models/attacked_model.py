import os.path
import numpy as np
from PIL import Image
from numpy import single
from typing import Tuple, Any


class AttackModel:
    def __init__(self, attacked_code_path, N, alpha, gamma, q=3):
        self.q = q
        self.N = N
        self.alpha = alpha
        self.gamma = gamma
        self.attacked_code_path = attacked_code_path

    def generate_MSG(self, data: list = None, index: int = 0) -> Tuple[Any, None]:
        file_path = os.path.join(self.attacked_code_path, f"{(index + 1):04d}.png")
        if os.path.exists(file_path):
            attacked_code = np.float32(Image.open(file_path))
            return attacked_code, None
        else:
            return None, None
