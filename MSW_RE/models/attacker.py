import cv2
import torch
import numpy as np
from PIL import Image
from numpy import ndarray
from scipy.ndimage import gaussian_filter
from skimage.util import img_as_float, view_as_windows
from torchvision import transforms
from experiments.attack.train_attack_model.unet import UNet


def _vote_downsample(image, symbol_size=2, thr=0.5):
    """An input image binarization with guaranteed integrity of each symbol

    Args:
        image: code to process
        symbol_size: size of the symbols' blocks in an input image
        thr: binarization threshold

    Returns:
        np.ndarray: binarized code"""

    symbols = view_as_windows(image, window_shape=symbol_size, step=symbol_size)
    symbols = symbols.reshape(symbols.shape[0], symbols.shape[1], -1)
    symbols = np.mean(symbols, axis=2)
    symbols[symbols < thr] = 0
    symbols[symbols != 0] = 1
    return symbols


class NetAttack:
    def __init__(self, model_path: str = None, print_dpi: int = 600, scan_ppi: int = 1200):
        """
        Initializes the NetAttack class for image restoration.

        This class simulates an attack on a scanned binary document,
        aiming to restore the original printed binary image.

        :param model_path: Path to the trained PyTorch model file (.pt or .pth).
        :param print_dpi: DPI of the original printed image.
        :param scan_ppi: PPI (Pixels Per Inch) of the scanned image.
        """
        self.v = scan_ppi // print_dpi  # Downsample factor based on resolution ratio
        self.model_path = model_path  # Model path
        self.Net = UNet(1, 1)  # Assuming a U-Net-like architecture for restoration
        self.Net.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.Net.eval()  # Set network to evaluation mode

        # Define preprocessing transform
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __call__(self, scanned_code: np.ndarray) -> np.ndarray:
        orig_h, orig_w = scanned_code.shape[:2]

        def round_up_to_multiple(x, base=32):
            return ((x + base - 1) // base) * base

        padded_h = round_up_to_multiple(orig_h)
        padded_w = round_up_to_multiple(orig_w)

        pad_bottom = padded_h - orig_h
        pad_right = padded_w - orig_w

        scanned_code_padded = np.pad(
            scanned_code,
            ((0, pad_bottom), (0, pad_right)),
            mode='edge'
        )

        tensor_scanned_code = self.transform(Image.fromarray(np.uint8(scanned_code_padded))).unsqueeze(0)

        with torch.no_grad():
            attacked_code = self.Net(tensor_scanned_code).squeeze(0).squeeze(0).detach().cpu().numpy()

        attacked_code_cropped = attacked_code[:orig_h, :orig_w]
        digital_attacked_code = np.round(attacked_code_cropped)
        voted_code = _vote_downsample(digital_attacked_code, self.v)
        return np.uint8(voted_code * 255.)


class BinAttack:
    def __init__(self, print_dpi: int = 600, scan_ppi: int = 1200):
        """
        Initializes the BinAttack class for image restoration.

        This class simulates an attack on a scanned document,
        aiming to restore the original printed binary image.

        :param print_dpi: DPI of the original printed image.
        :param scan_ppi: PPI (Pixels Per Inch) of the scanned image.
        """
        self.v = scan_ppi // print_dpi  # Downsample factor based on resolution ratio

    def _unsharp_mask_single_channel(self, image, radius, amount, vrange):
        """
        Applies unsharp masking to a single-channel image.

        Unsharp masking enhances edges by subtracting a blurred version of the image from itself.

        :param image: Input grayscale image as a NumPy array.
        :param radius: Radius for Gaussian blur kernel.
        :param amount: Strength of the sharpening effect.
        :param vrange: Value range [min, max] to clip final values. If None, no clipping.
        :return: Sharpened image as a NumPy array.
        """
        blurred = gaussian_filter(image, sigma=radius, mode='reflect')
        result = image + (image - blurred) * amount

        if vrange is not None:
            return np.clip(result, vrange[0], vrange[1], out=result)
        return result

    def unsharp_mask(self, image, radius=1.0, amount=1.0, multichannel=False,
                     preserve_range=False):
        """
        Applies unsharp mask to enhance sharpness of an image.

        Enhances sharpness using a high-pass filter-like technique.

        :param image: Input image (2D or 3D array).
        :param radius: Sigma value for Gaussian blur kernel.
        :param amount: Multiplier for the sharpening effect.
        :param multichannel: Whether input has multiple channels (e.g., RGB).
        :param preserve_range: Whether to keep original intensity range.
        :return: Sharpened image.
        """
        vrange = None  # Range for valid values; used for clipping.

        if preserve_range:
            fimg = image.astype(float)
        else:
            fimg = img_as_float(image)
            negative = np.any(fimg < 0)
            if negative:
                vrange = [-1., 1.]
            else:
                vrange = [0., 1.]

        if multichannel:
            result = np.empty_like(fimg, dtype=float)
            for channel in range(image.shape[-1]):
                result[..., channel] = self._unsharp_mask_single_channel(
                    fimg[..., channel], radius, amount, vrange)
            return result
        else:
            return self._unsharp_mask_single_channel(fimg, radius, amount, vrange)

    def binary(self, scanned: ndarray, unsharp_radius=2, unsharp_amount=4) -> ndarray:
        """
        Converts a scanned image into a binary representation.

        The process includes:
        1. Applying unsharp mask to enhance contrast
        2. Converting to binary using Otsu's thresholding

        :param scanned: Input scanned image as a NumPy array (grayscale expected).
        :param unsharp_radius: Radius for Gaussian blur in unsharp mask.
        :param unsharp_amount: Strength multiplier for sharpening.
        :return: Binary image with values 0 (black) and 255 (white).
        """
        scan = self.unsharp_mask(scanned, unsharp_radius, unsharp_amount)
        ret, binary_img = cv2.threshold(np.uint8(scan * 255), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary_img / 255.

    def __call__(self, scanned_code: ndarray, unsharp_radius=2, unsharp_amount=4) -> ndarray:
        """
        Main entry point of the BinAttack class.

        Takes a scanned image and performs:
        1. Image binarization using unsharp mask + Otsu thresholding
        2. Voting-based downsampling to recover the original binary image

        :param scanned_code: Scanned image matrix (H x W), assumed to be grayscale.
        :return: Restored binary image after processing.
        """
        digital_attacked_code = self.binary(scanned_code)
        voted_code = _vote_downsample(digital_attacked_code, self.v)
        return np.uint8(voted_code * 255.)
