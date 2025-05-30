import os

import bchlib
import cv2
import torch
import random
import string
import PyPDF2
import numpy as np
from PIL import Image
from typing import Tuple
from numpy import ndarray
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from scipy.ndimage import gaussian_filter
from reportlab.lib.utils import ImageReader
from skimage import img_as_float
from torchvision.transforms import transforms
from tqdm import tqdm


def generate_random_string(n):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for _ in range(n))
    return random_string


def get_string_difference(str1, str2):
    set1 = set(str1)
    set2 = set(str2)
    difference_set = set1.symmetric_difference(set2)
    difference_str = ''.join(sorted(difference_set))
    return difference_str


def find_all_files(data_dir, file_extensions=None):
    if file_extensions is None:
        file_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.bmp']
    image_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in file_extensions):
                image_files.append(os.path.abspath(os.path.join(root, file)))
    return image_files


class BCH:
    def __init__(self, BCH_POLYNOMIAL_=487, BCH_BITS_=5):
        self.bch = bchlib.BCH(BCH_POLYNOMIAL_, BCH_BITS_)

    def Encode(self, data_: bytearray):
        ecc = self.bch.encode(data_)
        packet = data_ + ecc
        packet_binary = ''.join(format(x, '08b') for x in packet)
        secret_ = [int(x) for x in packet_binary]
        return secret_

    def Decode(self, secret_: list):
        packet_binary = "".join([str(int(bit)) for bit in secret_])
        packet = bytes(int(packet_binary[i: i + 8], 2) for i in range(0, len(packet_binary), 8))
        packet = bytearray(packet)
        data_, ecc = packet[:-self.bch.ecc_bytes], packet[-self.bch.ecc_bytes:]
        bit_flips = self.bch.decode(data_, ecc)
        if bit_flips[0] != -1:
            return bit_flips[1]
        return None



class Workflow:
    def __init__(self):
        self.pdf_list = []
        pass

    def generate(self, save_dir: str, Num: int, model, data_len: int, N: int, alpha: float, gamma: float,
                 print_dpi: int = 600, inner_border_mm: int = 5, out_border_mm: int = 8, outer_border_mm: int = 10,
                 paper_height_mm: int = 297, paper_width_mm: int = 210):
        """
        Generate multiple papers (or images) based on specified parameters and save them.

        Args:
            save_dir (str): Directory to save generated papers.
            Num (int): Number of papers to generate.
            model: Model used for generating papers.
            data_len (int): Length of data related to the model.
            alpha (float): Alpha parameter.
            gamma (float): Gamma parameter.
            N (int): Mode of generation.
            print_dpi (int, optional): Print DPI. Defaults to 600.
            inner_border_mm (int, optional): Inner border size in millimeters. Defaults to 5.
            out_border_mm (int, optional): Outer border size in millimeters. Defaults to 8.
            outer_border_mm (int, optional): Outermost border size in millimeters. Defaults to 10.
            paper_height_mm (int, optional): Height of the paper in millimeters. Defaults to 297.
            paper_width_mm (int, optional): Width of the paper in millimeters. Defaults to 210.

        Returns:
            None
        """
        assert Num > 0
        assert print_dpi >= 300
        assert model is not None

        # Create directories if they do not exist
        os.makedirs(f"{save_dir}/digital_outer/", exist_ok=True)
        os.makedirs(f"{save_dir}/digital_located/", exist_ok=True)
        os.makedirs(f"{save_dir}/digital_inner/", exist_ok=True)
        os.makedirs(f"{save_dir}/digital_match/N_{N}/alpha_{alpha}/gamma_{gamma}", exist_ok=True)
        os.makedirs(f"{save_dir}/scanned_match/N_{N}/alpha_{alpha}/gamma_{gamma}", exist_ok=True)
        os.makedirs(f"{save_dir}/scanned_org/", exist_ok=True)
        os.makedirs(f"{save_dir}/scanned_located/", exist_ok=True)

        # Convert border sizes from millimeters to pixels based on print DPI
        inner_border_size = self.dpi2pixels(print_dpi, inner_border_mm)
        out_border_size = self.dpi2pixels(print_dpi, out_border_mm)
        outer_border_size = self.dpi2pixels(print_dpi, outer_border_mm)

        # Convert paper dimensions from millimeters to pixels based on print DPI
        paper_width = self.dpi2pixels(print_dpi, paper_width_mm)
        paper_height = self.dpi2pixels(print_dpi, paper_height_mm)

        msw_index = 0
        img_index = 0
        for _ in tqdm(range(Num),
                      desc=f"Generate MSW for parameters (N:{model.N}, alpha:{alpha}, gamma:{gamma}, img_index:{img_index})",
                      leave=False):
            # Generate inner paper layout
            inner_paper, img_index = self.layout_inner_paper(img_index, model, data_len,
                                                             inner_border_size, paper_height, paper_width,
                                                             out_border_size, outer_border_size)

            # Generate outer paper layout
            out_paper = self.layout_out_paper(inner_paper, print_dpi, paper_height, paper_width, outer_border_size,
                                              out_border_size)

            # Generate final printed paper layout
            printed_paper = self.layout_outer_paper(out_paper, paper_height, paper_width, outer_border_size,
                                                    msw_index + 1)

            # Save generated images and convert them to PDFs
            cv2.imwrite(
                f"{save_dir}/digital_outer/dpi_{print_dpi}_{msw_index + 1:04d}_{model.N}_{alpha}_{gamma}.png",
                np.uint8(printed_paper * 255))
            cv2.imwrite(
                f"{save_dir}/digital_located/dpi_{print_dpi}_{msw_index + 1:04d}_{model.N}_{alpha}_{gamma}.png",
                np.uint8(out_paper * 255))
            cv2.imwrite(
                f"{save_dir}/digital_inner/dpi_{print_dpi}_{msw_index + 1:04d}_{model.N}_{alpha}_{gamma}.png",
                np.uint8(inner_paper * 255))

            # Convert outer paper image to PDF
            convert_png_to_pdf_lossless(
                f"{save_dir}/digital_outer/dpi_{print_dpi}_{msw_index + 1:04d}_{model.N}_{alpha}_{gamma}.png",
                f"{save_dir}/digital_outer/dpi_{print_dpi}_{msw_index + 1:04d}_{model.N}_{alpha}_{gamma}.pdf")

            # Add PDF path to list
            self.pdf_list.append(
                f"{save_dir}/digital_outer/dpi_{print_dpi}_{msw_index + 1:04d}_{model.N}_{alpha}_{gamma}.pdf")

            msw_index += 1

    def generate_for_attack(self, Num: int, save_dir: str, save_captured: str, load_dir: str, print_dpi: int = 600,
                            inner_border_mm: int = 5, out_border_mm: int = 8, outer_border_mm: int = 10,
                            paper_height_mm: int = 297, paper_width_mm: int = 210):
        """
        Generate various attack scenarios for documents, save transformed images and PDFs.

        :param Num:
        :param save_dir: Directory to save the generated attack scenarios.
        :param save_captured: Directory to save captured images for the attack.
        :param load_dir: Directory containing the original images to process.
        :param print_dpi: DPI of the printed documents.
        :param inner_border_mm: Size of the inner border in millimeters.
        :param out_border_mm: Size of the outer border within the paper in millimeters.
        :param outer_border_mm: Size of the outer border outside the paper in millimeters.
        :param paper_height_mm: Height of the paper in millimeters.
        :param paper_width_mm: Width of the paper in millimeters.
        """
        assert print_dpi >= 300

        # Create directories for saving attack scenarios
        os.makedirs(f"{save_dir}/attack_outer/attack_identity/", exist_ok=True)
        os.makedirs(f"{save_dir}/attack_outer/attack_binary/", exist_ok=True)
        os.makedirs(f"{save_dir}/attack_outer/attack_network/", exist_ok=True)

        os.makedirs(f"{save_dir}/attack_located/attack_identity/", exist_ok=True)
        os.makedirs(f"{save_dir}/attack_located/attack_binary/", exist_ok=True)
        os.makedirs(f"{save_dir}/attack_located/attack_network/", exist_ok=True)

        os.makedirs(f"{save_dir}/attack_inner/attack_identity/", exist_ok=True)
        os.makedirs(f"{save_dir}/attack_inner/attack_binary/", exist_ok=True)
        os.makedirs(f"{save_dir}/attack_inner/attack_network/", exist_ok=True)

        os.makedirs(f"{save_captured}/attack_identity/", exist_ok=True)
        os.makedirs(f"{save_captured}/attack_binary/", exist_ok=True)
        os.makedirs(f"{save_captured}/attack_network/", exist_ok=True)

        os.makedirs(f"{save_captured.replace('attack_captured_all', 'attack_phone_all')}/attack_identity/",
                    exist_ok=True)
        os.makedirs(f"{save_captured.replace('attack_captured_all', 'attack_phone_all')}/attack_binary/", exist_ok=True)
        os.makedirs(f"{save_captured.replace('attack_captured_all', 'attack_phone_all')}/attack_network/",
                    exist_ok=True)

        # Convert millimeter dimensions to pixel dimensions
        inner_border_size = self.dpi2pixels(print_dpi, inner_border_mm)
        out_border_size = self.dpi2pixels(print_dpi, out_border_mm)
        outer_border_size = self.dpi2pixels(print_dpi, outer_border_mm)
        paper_width, paper_height = self.dpi2pixels(print_dpi, paper_width_mm), self.dpi2pixels(print_dpi,
                                                                                                paper_height_mm)
        all_image_list = find_all_files(load_dir)
        all_image_list.sort()
        pape_index = 0
        img_index = 0
        with tqdm(total=Num, desc="Processing Images") as pbar:
            while img_index < len(all_image_list) // 3:
                img_index, inner_paper_identity, inner_paper_binary, inner_paper_network = self.layout_inner_paper_for_attack(
                    img_index,
                    load_dir,
                    inner_border_size,
                    paper_height,
                    paper_width,
                    out_border_size,
                    outer_border_size)

                # Layout outer paper for identity attack
                out_paper_identity = self.layout_out_paper(inner_paper_identity, print_dpi, paper_height, paper_width,
                                                           outer_border_size, out_border_size)
                outer_paper_identity = self.layout_outer_paper(out_paper_identity, paper_height, paper_width,
                                                               outer_border_size, pape_index + 1)

                # Save outer paper identity attack scenario
                cv2.imwrite(f"{save_dir}/attack_outer/attack_identity/{pape_index + 1:04d}.png",
                            np.uint8(outer_paper_identity * 255))
                cv2.imwrite(f"{save_dir}/attack_located/attack_identity/{pape_index + 1:04d}.png",
                            np.uint8(out_paper_identity * 255))
                cv2.imwrite(f"{save_dir}/attack_inner/attack_identity/{pape_index + 1:04d}.png",
                            np.uint8(inner_paper_identity * 255))
                convert_png_to_pdf_lossless(f"{save_dir}/attack_outer/attack_identity/{pape_index + 1:04d}.png",
                                            f"{save_dir}/attack_outer/attack_identity/{pape_index + 1:04d}.pdf")

                # Layout outer paper for binary attack
                out_paper_binary = self.layout_out_paper(inner_paper_binary, print_dpi, paper_height, paper_width,
                                                         outer_border_size, out_border_size)
                outer_paper_binary = self.layout_outer_paper(out_paper_binary, paper_height, paper_width,
                                                             outer_border_size,
                                                             pape_index + 1)

                # Save outer paper binary attack scenario
                cv2.imwrite(f"{save_dir}/attack_outer/attack_binary/{pape_index + 1:04d}.png",
                            np.uint8(outer_paper_binary * 255))
                cv2.imwrite(f"{save_dir}/attack_located/attack_binary/{pape_index + 1:04d}.png",
                            np.uint8(out_paper_binary * 255))
                cv2.imwrite(f"{save_dir}/attack_inner/attack_binary/{pape_index + 1:04d}.png",
                            np.uint8(inner_paper_binary * 255))
                convert_png_to_pdf_lossless(f"{save_dir}/attack_outer/attack_binary/{pape_index + 1:04d}.png",
                                            f"{save_dir}/attack_outer/attack_binary/{pape_index + 1:04d}.pdf")

                # Layout outer paper for network attack
                out_paper_network = self.layout_out_paper(inner_paper_network, print_dpi, paper_height, paper_width,
                                                          outer_border_size, out_border_size)
                outer_paper_network = self.layout_outer_paper(out_paper_network, paper_height, paper_width,
                                                              outer_border_size,
                                                              pape_index + 1)

                # Save outer paper network attack scenario
                cv2.imwrite(f"{save_dir}/attack_outer/attack_network/{pape_index + 1:04d}.png",
                            np.uint8(outer_paper_network * 255))
                cv2.imwrite(f"{save_dir}/attack_located/attack_network/{pape_index + 1:04d}.png",
                            np.uint8(out_paper_network * 255))
                cv2.imwrite(f"{save_dir}/attack_inner/attack_network/{pape_index + 1:04d}.png",
                            np.uint8(inner_paper_network * 255))
                convert_png_to_pdf_lossless(f"{save_dir}/attack_outer/attack_network/{pape_index + 1:04d}.png",
                                            f"{save_dir}/attack_outer/attack_network/{pape_index + 1:04d}.pdf")

                pape_index += 1

                # 更新进度条
                pbar.update(1)

    def calculate_angle_with_horizontal(self, pointA, pointB):
        """
        Calculate the angle between two points (pointA and pointB) and the horizontal axis.

        Args:
            pointA (tuple): Coordinates of point A (y, x).
            pointB (tuple): Coordinates of point B (y, x).

        Returns:
            float: Angle in degrees between the line connecting pointA and pointB and the horizontal axis.
        """
        x1, y1 = pointA
        x2, y2 = pointB
        dx = x2 - x1
        dy = y2 - y1
        theta_radians = np.arctan2(dy, dx)
        theta_degrees = np.degrees(theta_radians)
        return theta_degrees

    def rotated_image(self, captured_img, theta_degrees):
        """
        Rotate an image by a specified angle.

        Args:
            captured_img (np.ndarray): Image to be rotated.
            theta_degrees (float): Angle of rotation in degrees.

        Returns:
            np.ndarray: Rotated image.
        """
        (h, w) = captured_img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, -theta_degrees, 1.0)
        rotated_image = cv2.warpAffine(captured_img, M, (w, h))
        return rotated_image

    def iteratively_locate_corner_points(self, captured_img, resize_shape, scan_ppi, iteration: int = 5):
        """
        Iteratively locate corner points of an image.

        Args:
            captured_img (np.ndarray): Image to be processed.
            resize_shape (tuple): Shape to resize the image to (height, width).
            scan_ppi (int): Scan resolution in pixels per inch (PPI).
            iteration (int, optional): Number of iterations. Defaults to 5.

        Returns:
            tuple: Tuple containing:
                - np.ndarray: Processed image after iterations.
                - list: Adjusted coordinates of the corner points in the order of top-left, top-right, bottom-right, bottom-left.
                - np.ndarray: Edge-detected image.
        """
        global adjust_pts, edges
        assert iteration >= 1

        for _ in range(iteration):
            img_for_location = cv2.resize(captured_img, dsize=resize_shape)
            location_pts = self.locate_edge_points(img_for_location)
            original_location_pts = self.scale_points_to_original(location_pts, captured_img.shape,
                                                                  img_for_location.shape)
            adjust_pts, edges = self.adjust_edge_points(captured_img, original_location_pts, scan_ppi)
            theta_degree = self.calculate_angle_with_horizontal(adjust_pts[0], adjust_pts[3])
            captured_img = self.rotated_image(captured_img, theta_degree)

        # Example: Further rotation adjustment (modify as needed)
        captured_img = self.rotated_image(captured_img, 0.2)  # Additional rotation
        return captured_img, adjust_pts, edges

    def locate_contours(self, save_dir: str, load_dir: str, print_dpi: int = 600, scan_ppi: int = 1200,
                        outer_border_mm: int = 10, paper_height_mm: int = 297, paper_width_mm: int = 210):
        """
        Locate contours of scanned documents, transform perspective, and save the transformed images.

        :param save_dir: Directory to save the transformed images.
        :param load_dir: Directory containing the scanned images.
        :param print_dpi: DPI of the printed documents.
        :param scan_ppi: PPI of the scanned images.
        :param outer_border_mm: Size of the outer border in millimeters.
        :param paper_height_mm: Height of the paper in millimeters.
        :param paper_width_mm: Width of the paper in millimeters.
        """
        # Create the save directory if it doesn't exist
        os.makedirs(f"{save_dir}/", exist_ok=True)

        # Convert paper dimensions and outer border size from millimeters to pixels
        paper_width = self.dpi2pixels(print_dpi, paper_width_mm)
        paper_height = self.dpi2pixels(print_dpi, paper_height_mm)
        outer_border_size = self.dpi2pixels(print_dpi, outer_border_mm)

        # Calculate the internal paper dimensions (excluding borders)
        out_paper_height, out_paper_width = paper_height - 2 * outer_border_size, paper_width - 2 * outer_border_size

        # Calculate the ratio between scan PPI and print DPI
        v = scan_ppi // print_dpi

        # Calculate the saved paper size (adjusted according to the scan-to-print ratio)
        save_size = (out_paper_width * v, out_paper_height * v)

        # Get a list of all image files to process
        captured_dir_list = find_all_files(load_dir)
        captured_dir_list.sort()

        # Set resize dimensions for corner point location iteration
        resize_width = self.dpi2pixels(150, paper_width_mm)
        resize_height = self.dpi2pixels(150, paper_height_mm)

        index_save = 0

        # Process each image
        for img_dir in tqdm(captured_dir_list, desc="locate the corner points of the captured MSW:"):
            # Read the image in grayscale mode
            captured_img = cv2.imdecode(np.fromfile(img_dir, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

            # Iteratively locate corner points
            captured_img, adjust_pts, edges = self.iteratively_locate_corner_points(captured_img,
                                                                                    (resize_width, resize_height),
                                                                                    scan_ppi)

            # Perform perspective transformation
            dst_points = [(0, 0), (0, save_size[0]), (save_size[1], save_size[0]), (save_size[1], 0)]
            src_points = np.array(adjust_pts, dtype=np.float32)
            dst_points = np.array(dst_points, dtype=np.float32)
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            trans_img = cv2.warpPerspective(captured_img, matrix, save_size)

            # Save the transformed image
            cv2.imwrite(f'{save_dir}/{index_save + 1:04d}.png', trans_img)
            index_save += 1

    def calculate_ppi(self, captured_img, paper_height_mm, paper_width_mm):
        """
        Calculate the scanning PPI (pixels per inch) based on the captured image and paper dimensions.

        :param captured_img: The captured image
        :param paper_height_mm: The height of the paper in millimeters
        :param paper_width_mm: The width of the paper in millimeters
        :return: The calculated PPI value
        """
        # Convert paper dimensions from millimeters to inches
        paper_height_inches = paper_height_mm / 25.4
        paper_width_inches = paper_width_mm / 25.4

        # Get the resolution of the captured image
        img_height, img_width = captured_img.shape[:2]

        # Calculate the PPI for both height and width
        ppi_height = img_height / paper_height_inches
        ppi_width = img_width / paper_width_inches

        # Average the PPI values from both directions
        ppi = (ppi_height + ppi_width) / 2

        return ppi

    def locate_contours_phone(self, save_dir: str, load_dir: str, print_dpi: int = 600, scan_ppi: int = 1200,
                              outer_border_mm: int = 10, paper_height_mm: int = 297, paper_width_mm: int = 210):
        """
        Locate contours of scanned documents, transform perspective, and save the transformed images.

        :param save_dir: Directory to save the transformed images.
        :param load_dir: Directory containing the scanned images.
        :param print_dpi: DPI of the printed documents.
        :param scan_ppi: PPI of the scanned images.
        :param outer_border_mm: Size of the outer border in millimeters.
        :param paper_height_mm: Height of the paper in millimeters.
        :param paper_width_mm: Width of the paper in millimeters.
        """
        # Create the save directory if it doesn't exist
        os.makedirs(f"{save_dir}/", exist_ok=True)

        # Convert paper dimensions and outer border size from millimeters to pixels
        paper_width = self.dpi2pixels(print_dpi, paper_width_mm)
        paper_height = self.dpi2pixels(print_dpi, paper_height_mm)
        outer_border_size = self.dpi2pixels(print_dpi, outer_border_mm)

        # Calculate the internal paper dimensions (excluding borders)
        out_paper_height, out_paper_width = paper_height - 2 * outer_border_size, paper_width - 2 * outer_border_size

        # Calculate the ratio between scan PPI and print DPI
        v = scan_ppi // print_dpi

        # Calculate the saved paper size (adjusted according to the scan-to-print ratio)
        save_size = (out_paper_width * v, out_paper_height * v)

        # Get a list of all image files to process
        captured_dir_list = find_all_files(load_dir)
        captured_dir_list.sort()
        print(load_dir, save_dir, captured_dir_list)
        # Set resize dimensions for corner point location iteration
        resize_width = self.dpi2pixels(150, paper_width_mm)
        resize_height = self.dpi2pixels(150, paper_height_mm)

        index_save = 0

        # Process each image
        for img_dir in tqdm(captured_dir_list, desc="locate the corner points of the captured MSW:"):
            # Read the image in grayscale mode
            captured_img = cv2.imdecode(np.fromfile(img_dir, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            captured_ppi = self.calculate_ppi(captured_img, paper_height_mm, paper_width_mm)
            # Iteratively locate corner points
            captured_img, adjust_pts, edges = self.iteratively_locate_corner_points(captured_img,
                                                                                    (resize_width, resize_height),
                                                                                    captured_ppi)

            # Perform perspective transformation
            dst_points = [(0, 0), (0, save_size[0]), (save_size[1], save_size[0]), (save_size[1], 0)]
            src_points = np.array(adjust_pts, dtype=np.float32)
            dst_points = np.array(dst_points, dtype=np.float32)
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            trans_img = cv2.warpPerspective(captured_img, matrix, save_size)

            # Save the transformed image
            cv2.imwrite(f'{save_dir}/{index_save + 1:04d}.png', trans_img)
            index_save += 1

    def scale_points_to_original(self, location_pts, original_shape, resize_shape):
        """
        Scale points from a resized image back to their coordinates in the original image.

        Args:
            location_pts (list): List of points in the resized image [(y_s1, x_s1), (y_s2, x_s2), ...].
            original_shape (tuple): Shape of the original image (height, width).
            resize_shape (tuple): Shape of the resized image (height_s, width_s).

        Returns:
            list: List of points in the original image [(y1, x1), (y2, x2), ...].
        """
        H, W = original_shape
        Hs, Ws = resize_shape
        scaled_points = []
        for (y_s, x_s) in location_pts:
            y = int(y_s / (Hs / H))  # Scale y-coordinate back to original size
            x = int(x_s / (Ws / W))  # Scale x-coordinate back to original size
            scaled_points.append((y, x))
        return scaled_points

    def adjust_edge_points(self, img_for_location, located_pts: list, scan_ppi: int = 1200, crop_size_mm: int = 5):
        """
        Adjust the four corner edge points of the border.

        Args:
            img_for_location (np.ndarray): Image for locating edge points.
            located_pts (list): Coordinates of the located edge points in the order of top-left, top-right, bottom-right, bottom-left.
            scan_ppi (int): Scan resolution in pixels per inch (PPI).
            crop_size_mm (int): Size in millimeters for cropping.

        Returns:
            tuple: Tuple containing:
                - list: Adjusted coordinates of the four edge points in the order of top-left, top-right, bottom-right, bottom-left.
                - np.ndarray: Edge-detected image.
        """

        # 1. Edge detection
        kernel = np.ones((3, 3), np.uint8)  # You can adjust the size and shape of the kernel
        img_for_location = cv2.dilate(img_for_location, kernel, iterations=1)
        img_for_location = cv2.erode(img_for_location, kernel, iterations=1)
        _, edges = cv2.threshold(img_for_location, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = 255 - edges

        # 2. Unit conversion
        crop_size = self.dpi2pixels(scan_ppi, crop_size_mm)

        # 3. Calculate reference points
        h, w = img_for_location.shape[:2]
        reference_points = [
            (located_pts[0][0] - crop_size // 2, located_pts[0][1] - crop_size // 2),  # Top-left
            (located_pts[1][0] - crop_size // 2, located_pts[1][1] + crop_size // 2),  # Top-right
            (located_pts[2][0] + crop_size // 2, located_pts[2][1] + crop_size // 2),  # Bottom-right
            (located_pts[3][0] + crop_size // 2, located_pts[3][1] - crop_size // 2)  # Bottom-left
        ]

        # 4. Initialize traversal points
        center_points = located_pts

        # 5. Find nearest edge points and save the edge-detected image
        edge_points = []
        for idx, ((ref_y, ref_x), (center_y, center_x)) in enumerate(zip(reference_points, center_points)):
            min_dist = float('inf')
            edge_point = (center_y, center_x)
            for y in range(-crop_size // 2, crop_size // 2):
                for x in range(-crop_size // 2, crop_size // 2):
                    loc_y = center_y + y
                    loc_x = center_x + x
                    if h > loc_y >= 0 != edges[loc_y, loc_x] and 0 <= loc_x < w:  # Found edge point
                        dist = np.sqrt((loc_y - ref_y) ** 2 + (loc_x - ref_x) ** 2)
                        if dist < min_dist:
                            min_dist = dist
                            edge_point = (loc_y, loc_x)
            edge_points.append(edge_point)

        # Example: Compensate edge points (adjust as per your specific needs)
        compensated_pts = [(-0, 0), (-0, 0), (-0, 0), (-0, 0)]
        final_pts = [(edge_point[0] + compensated_pt[0], edge_point[1] + compensated_pt[1]) for
                     edge_point, compensated_pt in zip(edge_points, compensated_pts)]
        return final_pts, edges

    def locate_edge_points(self, img_for_location, border_crop_mm_h: list = [10, 10, 12, 12],
                           border_crop_mm_w: list = [10, 15, 15, 9], crop_size_mm: int = 10):
        """
        Locate the four corner edge points of the border.

        Args:
            img_for_location (np.ndarray): Image for locating edge points.
            border_crop_mm_h (list): Border widths in millimeters for height direction.
            border_crop_mm_w (list): Border widths in millimeters for width direction.
            crop_size_mm (int): Size in millimeters for cropping.

        Returns:
            list: Coordinates of the four edge points in the order of top-left, top-right, bottom-right, bottom-left.
        """

        # 1. Edge detection
        kernel = np.ones((3, 3), np.uint8)  # Define a 3x3 kernel
        img_for_location = cv2.dilate(img_for_location, kernel, iterations=2)  # Erosion operation
        img_for_location = cv2.erode(img_for_location, kernel, iterations=2)  # Erosion operation
        _, edges = cv2.threshold(img_for_location, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = 255 - edges

        # 2. Unit conversion
        border_crop_h = [self.dpi2pixels(150, mm) for mm in border_crop_mm_h]
        border_crop_w = [self.dpi2pixels(150, mm) for mm in border_crop_mm_w]
        crop_size = self.dpi2pixels(150, crop_size_mm)

        # 3. Calculate reference points
        h, w = img_for_location.shape[:2]
        reference_points = [
            (border_crop_h[0], border_crop_w[0]),  # Top-left
            (border_crop_h[1], w - border_crop_w[1]),  # Top-right
            (h - border_crop_h[2], w - border_crop_w[2]),  # Bottom-right
            (h - border_crop_h[3], border_crop_w[3])  # Bottom-left
        ]

        # 4. Initialize traversal points
        center_points = [
            (border_crop_h[0] + crop_size // 2, border_crop_w[0] + crop_size // 2),  # Top-left
            (border_crop_h[1] + crop_size // 2, w - border_crop_w[1] - crop_size // 2),  # Top-right
            (h - border_crop_h[2] - crop_size // 2, w - border_crop_w[2] - crop_size // 2),  # Bottom-right
            (h - border_crop_h[3] - crop_size // 2, border_crop_w[3] + crop_size // 2)  # Bottom-left
        ]

        # 5. Find nearest edge points and save the edge image
        edge_points = []
        for idx, ((ref_y, ref_x), (center_y, center_x)) in enumerate(zip(reference_points, center_points)):
            min_dist = float('inf')
            edge_point = (center_y, center_x)
            for y in range(-crop_size // 2, crop_size // 2):
                for x in range(-crop_size // 2, crop_size // 2):
                    loc_y = center_y + y
                    loc_x = center_x + x
                    if h > loc_y >= 0 != edges[loc_y, loc_x] and 0 <= loc_x < w:  # Found edge point
                        dist = np.sqrt((loc_y - ref_y) ** 2 + (loc_x - ref_x) ** 2)
                        if dist < min_dist:
                            min_dist = dist
                            edge_point = (loc_y, loc_x)
            edge_points.append(edge_point)
        return edge_points

    def merge_and_delete_pdfs(self, output_path: str):
        """
        Merge all PDF files in the specified directory and delete the originals.

        Args:
            output_path (str): Directory path where PDF files are located.

        Returns:
            None
        """
        # Find all PDF files in the output_path directory
        pdf_list = find_all_files(output_path, file_extensions=['.pdf'])
        pdf_list.sort()  # Sort the PDF file list

        # Initialize PDF merger
        pdf_merger = PyPDF2.PdfMerger()

        # Merge all found PDF files
        for pdf in pdf_list:
            if os.path.exists(pdf):
                pdf_merger.append(pdf)  # Append each PDF file to the merger
            else:
                print(f"File not found: {pdf}")  # Print a message if a file is not found

        # Write the merged PDF to a new file named "all.pdf"
        with open(os.path.join(output_path, "all.pdf"), 'wb') as output_file:
            pdf_merger.write(output_file)

        # Close the PDF merger
        pdf_merger.close()

        # Delete the original PDF files
        for pdf in pdf_list:
            if os.path.exists(pdf):
                os.remove(pdf)  # Remove the PDF file
                print(f"Deleted file: {pdf}")  # Print a message that the file was deleted
            else:
                print(f"File not found, cannot delete: {pdf}")  # Print a message if a file is not found

    def layout_outer_paper(self, out_paper: np.ndarray, paper_height: int, paper_width: int, outer_border_size: int,
                           img_index: int = 0):
        """
        Generate layout for the final printed paper with outer paper embedded.

        Args:
            out_paper (np.ndarray): Outer paper layout as a numpy array.
            paper_height (int): Height of the final printed paper.
            paper_width (int): Width of the final printed paper.
            outer_border_size (int): Size of outer border.

        Returns:
            np.ndarray: Final printed paper layout as a numpy array with outer paper embedded.
        """
        # Initialize the final printed paper array
        printed_paper = np.ones(shape=(paper_height, paper_width))

        # Calculate positions to embed outer paper
        start_h = outer_border_size
        start_w = outer_border_size
        end_h = start_h + out_paper.shape[0]
        end_w = start_w + out_paper.shape[1]

        # Embed outer paper into the final printed paper
        printed_paper[start_h:end_h, start_w:end_w] = out_paper

        # Example: Adding a specific region for illustration purposes (remove or adjust as needed)
        printed_paper[20:40, 20:40] = 0
        if img_index % 2 == 0:
            printed_paper[25:35, 25:35] = 1
        return printed_paper

    def layout_out_paper(self, inner_paper: np.ndarray, print_dpi: int, paper_height: int, paper_width: int,
                         outer_border_size: int, out_border_size: int):
        """
        Generate layout for outer paper with inner paper embedded.

        Args:
            inner_paper (np.ndarray): Inner paper layout as a numpy array.
            print_dpi (int): DPI of the print.
            paper_height (int): Height of the outer paper.
            paper_width (int): Width of the outer paper.
            outer_border_size (int): Size of outer border.
            out_border_size (int): Size of outer border.

        Returns:
            np.ndarray: Outer paper layout as a numpy array with inner paper embedded.
        """
        # Calculate border width based on print DPI
        border_width = int((1 / 200.) * print_dpi)
        if border_width < 1:
            border_width = 1

        # Calculate outer paper dimensions
        out_paper_height = paper_height - 2 * outer_border_size
        out_paper_width = paper_width - 2 * outer_border_size
        out_paper_size = (out_paper_height, out_paper_width)

        # Initialize outer paper array
        out_paper = np.ones(shape=out_paper_size)

        # Calculate positions to embed inner paper
        start_h = out_border_size
        start_w = out_border_size
        end_h = start_h + inner_paper.shape[0]
        end_w = start_w + inner_paper.shape[1]

        # Embed inner paper into outer paper
        out_paper[start_h:end_h, start_w:end_w] = inner_paper

        # Apply borders around the outer paper
        out_paper[:border_width, :] = 0  # Top border
        out_paper[-border_width:, :] = 0  # Bottom border
        out_paper[:, :border_width] = 0  # Left border
        out_paper[:, -border_width:] = 0  # Right border

        return out_paper

    def layout_inner_paper_for_attack(self, img_index, load_dir, inner_border_size: int, paper_height: int,
                                      paper_width: int,
                                      out_border_size: int, outer_border_size: int):
        """
        Generate a layout for inner paper for attack simulation.

        Args:
            load_dir (str): Directory path containing images.
            inner_border_size (int): Inner border size.
            paper_height (int): Height of the paper.
            paper_width (int): Width of the paper.
            out_border_size (int): Outer border size.
            outer_border_size (int): Outer border size.

        Returns:
            tuple: Tuple containing three numpy arrays:
                - inner_paper_identity: Identity image layout.
                - inner_paper_binary: Binary image layout.
                - inner_paper_network: Network image layout.
                :param img_index:
        """
        v = 1
        # Calculate inner paper dimensions
        inner_paper_height = paper_height - 2 * outer_border_size - 2 * out_border_size
        inner_paper_width = paper_width - 2 * outer_border_size - 2 * out_border_size
        inner_paper_size = (inner_paper_height, inner_paper_width)

        # Initialize arrays for identity, binary, and network images
        inner_paper_identity = np.ones(shape=inner_paper_size)
        inner_paper_binary = np.ones(shape=inner_paper_size)
        inner_paper_network = np.ones(shape=inner_paper_size)

        # Load network image to determine size
        network_img_path = os.path.join(load_dir, f"0001_network.png")
        network_img_for_shape = cv2.imdecode(np.fromfile(network_img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        one_msw_size = (network_img_for_shape.shape[0] + 2 * inner_border_size,
                        network_img_for_shape.shape[1] + 2 * inner_border_size)

        # Iterate over the inner paper area to place images
        for h in range(0, inner_paper_size[0], one_msw_size[0]):
            for w in range(0, inner_paper_size[1], one_msw_size[1]):
                start_h = h
                start_w = w
                end_h = start_h + one_msw_size[0]
                end_w = start_w + one_msw_size[1]

                # Ensure within bounds
                if 0 <= end_h < inner_paper_size[0] and 0 <= end_w <= inner_paper_size[1]:
                    # Load identity, binary, and network images
                    identity_img_path = os.path.join(load_dir, f"{img_index + 1:04d}_identity.png")
                    binary_img_path = os.path.join(load_dir, f"{img_index + 1:04d}_binary.png")
                    network_img_path = os.path.join(load_dir, f"{img_index + 1:04d}_network.png")

                    identity_img = cv2.imdecode(np.fromfile(identity_img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    identity_img = cv2.resize(identity_img, dsize=network_img_for_shape.shape[:2])
                    identity_marked_code = np.pad(identity_img / 255, pad_width=v, mode='constant', constant_values=0)
                    pad_identity_code = np.pad(identity_marked_code, pad_width=inner_border_size - v, mode='constant',
                                               constant_values=1)
                    inner_paper_identity[start_h:end_h, start_w:end_w] = pad_identity_code

                    binary_img = cv2.imdecode(np.fromfile(binary_img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    binary_img = cv2.resize(binary_img, dsize=network_img_for_shape.shape[:2],
                                            interpolation=cv2.INTER_NEAREST)
                    binary_marked_code = np.pad(binary_img / 255, pad_width=1, mode='constant', constant_values=0)
                    pad_binary_code = np.pad(binary_marked_code, pad_width=inner_border_size - 1, mode='constant',
                                             constant_values=1)
                    inner_paper_binary[start_h:end_h, start_w:end_w] = pad_binary_code

                    network_img = cv2.imdecode(np.fromfile(network_img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    network_img = cv2.resize(network_img, dsize=network_img_for_shape.shape[:2],
                                             interpolation=cv2.INTER_NEAREST)
                    network_marked_code = np.pad(network_img / 255, pad_width=v, mode='constant', constant_values=0)
                    pad_network_code = np.pad(network_marked_code, pad_width=inner_border_size - v, mode='constant',
                                              constant_values=1)
                    inner_paper_network[start_h:end_h, start_w:end_w] = pad_network_code
                    img_index += 1
        return img_index, inner_paper_identity, inner_paper_binary, inner_paper_network

    def layout_inner_paper(self, img_index, model, data_len: int,
                           inner_border_size: int,
                           paper_height: int, paper_width: int, out_border_size: int, outer_border_size: int):
        """
        Generate the layout for the inner paper.

        Args:
            img_index (int): Current image index.
            model: Model used for generating marked codes.
            data_len (int): Length of data related to the model.
            inner_border_size (int): Size of the inner border.
            paper_height (int): Height of the paper.
            paper_width (int): Width of the paper.
            out_border_size (int): Outer border size.
            outer_border_size (int): Outermost border size.

        Returns:
            ndarray: Generated inner paper layout.
            int: Updated image index.
        """
        # Calculate inner paper dimensions
        inner_paper_height = paper_height - 2 * outer_border_size - 2 * out_border_size
        inner_paper_width = paper_width - 2 * outer_border_size - 2 * out_border_size
        inner_paper_size = (inner_paper_height, inner_paper_width)

        # Initialize inner paper with ones (white background)
        inner_paper = np.ones(shape=inner_paper_size)

        # Generate marked code based on mode and parameters

        # Adjust marked code for "cdp" mode
        bits = [random.randint(0, model.q - 1) for _ in range(data_len)]
        marked_code, _ = model.generate_MSG(bits, index=img_index)
        # Calculate size of one marked code section including borders
        one_msw_size = (marked_code.shape[0] + 2 * inner_border_size, marked_code.shape[1] + 2 * inner_border_size)

        # Iterate over possible positions to place marked codes on inner paper
        for h in range(0, inner_paper_size[0], one_msw_size[0]):
            for w in range(0, inner_paper_size[1], one_msw_size[1]):
                start_h = h
                start_w = w
                end_h = start_h + one_msw_size[0]
                end_w = start_w + one_msw_size[1]

                # Check if the current position is within bounds of inner paper
                if 0 <= end_h < inner_paper_size[0] and 0 <= end_w <= inner_paper_size[1]:
                    # Pad marked code with border and place on inner paper
                    marked_code_ = np.pad(marked_code / 255, pad_width=1, mode='constant', constant_values=0)
                    pad_marked_code_ = np.pad(marked_code_, pad_width=inner_border_size - 1, mode='constant',
                                              constant_values=1)
                    inner_paper[start_h:end_h, start_w:end_w] = pad_marked_code_
                    # Update image index for next iteration
                    img_index += 1
                    print(img_index)
                    bits = [random.randint(0, model.q - 1) for _ in range(data_len)]
                    # Generate new marked code for next position if not "cdp" mode
                    marked_code, _ = model.generate_MSG(bits, index=img_index)
        return inner_paper, img_index

    def dpi2pixels(self, print_dpi: int, paper_length_mm: int):
        """
        Convert paper length from millimeters to pixels based on print DPI.

        Args:
            print_dpi (int): Dots per inch of the printer or display device.
            paper_length_mm (int): Length of the paper in millimeters.

        Returns:
            int: Length of the paper in pixels.
        """
        mm_to_inches = 1 / 25.4
        paper_inches = paper_length_mm * mm_to_inches
        img_size = int(paper_inches * print_dpi)
        return img_size

    def createDataset(self, digital_dir: str, captured_dir: str, save_dir: str, mgw_size: Tuple[int, int],
                      print_dpi: int = 600, scan_ppi: int = 1200, inner_border_mm: int = 5, out_border_mm: int = 8):
        """
        Process and align digital and captured images for creating a dataset.

        Parameters:
        - digital_dir (str): Directory containing the digital images.
        - captured_dir (str): Directory containing the captured images.
        - save_dir (str): Directory to save the processed images.
        - mgw_size (Tuple[int, int]): Size of the watermark grid.
        - print_dpi (int, optional): DPI value used for printing. Default is 600.
        - scan_ppi (int, optional): PPI value used for scanning. Default is 1200.
        - inner_border_mm (int, optional): Size of the inner border in millimeters. Default is 5.
        - out_border_mm (int, optional): Size of the outer border in millimeters. Default is 8.
        """

        v = scan_ppi // print_dpi
        msw_size = (int(mgw_size[0] * v), int(mgw_size[1] * v))
        inner_border_size, out_border_size = int(self.dpi2pixels(print_dpi, inner_border_mm) * v), int(
            self.dpi2pixels(print_dpi, out_border_mm) * v)
        digital_out_match_dir = f"{digital_dir}/"
        captured_out_dir = f"{captured_dir}/"
        os.makedirs(f"{save_dir.replace('genuine', 'digital')}", exist_ok=True)
        os.makedirs(f"{save_dir}", exist_ok=True)
        digital_match_img_list = find_all_files(digital_out_match_dir)
        captured_match_img_list = find_all_files(captured_out_dir)
        digital_match_img_list.sort()
        captured_match_img_list.sort()
        img_index = 0

        for digital_dir, captured_dir in tqdm(zip(digital_match_img_list, captured_match_img_list),
                                              total=len(digital_match_img_list),
                                              desc="create the dataset by the located MSW"):
            digital_match_img = cv2.imdecode(np.fromfile(digital_dir, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            digital_match_img = cv2.resize(digital_match_img, dsize=(
                int(digital_match_img.shape[1] * v), int(digital_match_img.shape[0] * v)),
                                           interpolation=cv2.INTER_NEAREST)
            captured_out_img = cv2.imdecode(np.fromfile(captured_dir, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            start_h = out_border_size
            start_w = out_border_size
            end_h = digital_match_img.shape[0] - out_border_size
            end_w = digital_match_img.shape[1] - out_border_size
            inner_digital_match_img = digital_match_img[start_h:end_h, start_w:end_w]
            inner_captured_out_img = captured_out_img[start_h:end_h, start_w:end_w]
            img_index = self.split_inner_paper(save_dir, img_index, inner_digital_match_img, inner_captured_out_img,
                                               inner_border_size, msw_size, mgw_size)

    def split_inner_paper(self, save_dir: str, img_index: int, inner_digital_match_img: np.ndarray,
                          inner_captured_out_img: np.ndarray, inner_border_size: int, msw_size: Tuple[int, int],
                          original_size: Tuple[int, int]) -> int:
        """
        Split the inner paper into smaller blocks and find the matched regions.

        :param save_dir: Directory to save the split images.
        :param img_index: Starting index for the image files.
        :param inner_digital_match_img: The inner digital image to be matched.
        :param inner_captured_out_img: The captured outer image containing the inner paper.
        :param inner_border_size: Size of the border around the matching window.
        :param msw_size: Size of the matching window (without borders).
        :param original_size: Original size to which the matched region should be resized.
        :return: Updated image index after processing.
        """
        # Calculate the size of one matching window block, including borders
        one_msw_size = (msw_size[0] + 2 * inner_border_size, msw_size[1] + 2 * inner_border_size)

        # Iterate over the height and width of the inner digital match image in steps of one_msw_size
        for h in range(0, inner_digital_match_img.shape[0], one_msw_size[0]):
            for w in range(0, inner_digital_match_img.shape[1], one_msw_size[1]):
                start_h = h
                start_w = w
                end_h = start_h + one_msw_size[0]
                end_w = start_w + one_msw_size[1]

                # Check if the block is within the bounds of the image
                if 0 <= end_h < inner_digital_match_img.shape[0] and 0 <= end_w <= inner_digital_match_img.shape[1]:
                    # Extract the block from both the digital match image and the captured image
                    digital_one_msw_block = inner_digital_match_img[start_h:end_h, start_w:end_w]
                    captured_one_msw_block = inner_captured_out_img[start_h:end_h, start_w:end_w]

                    # Extract the inner part of the block without borders
                    digital_one_msw = digital_one_msw_block[inner_border_size:inner_border_size + msw_size[0],
                                      inner_border_size:inner_border_size + msw_size[1]]

                    # Find the matched region using the template and corner method
                    matched_region_one = self.find_matched_images_by_template_and_corner(captured_one_msw_block,
                                                                                         digital_one_msw,
                                                                                         inner_border_size)
                    matched_region_two = self.find_matched_images_by_template(captured_one_msw_block, digital_one_msw)
                    corr_one = np.corrcoef(matched_region_one, digital_one_msw)[0, 1]
                    corr_two = np.corrcoef(matched_region_two, digital_one_msw)[0, 1]
                    if corr_one > corr_two:
                        matched_region = matched_region_one
                    else:
                        matched_region = matched_region_two

                    # Resize the digital matching window to the original size
                    digital_one_msw_save = cv2.resize(digital_one_msw, dsize=original_size,
                                                      interpolation=cv2.INTER_NEAREST)

                    # Save the digital matching window and the matched region
                    cv2.imwrite(f"{save_dir.replace('genuine', 'digital')}/{(img_index + 1):04d}.png",
                                np.uint8(digital_one_msw_save))
                    cv2.imwrite(f"{save_dir}/{(img_index + 1):04d}.png", np.uint8(matched_region))

                    # Increment the image index
                    img_index += 1

        # Return the updated image index
        return img_index

    def find_matched_images_by_template_and_corner(self, captured_img: np.ndarray, template: np.ndarray,
                                                   crop_size: int) -> np.ndarray:
        """
        Find the matched template in the captured image and extract the corners.

        :param captured_img: Captured image containing the template.
        :param template: The template image to be matched.
        :return: Transformed image using perspective transform based on the closest edge points.
        """
        # Get the height and width of the captured image
        captured_height, captured_width = captured_img.shape
        # Calculate reference points: top-left, top-right, bottom-right, and bottom-left
        # reference_points = [
        #     (crop_size // 2, crop_size // 2),  # Top-left
        #     (crop_size // 2, captured_width - crop_size // 2),  # Top-right
        #     (captured_height - crop_size // 2, captured_width - crop_size // 2),  # Bottom-right
        #     (captured_height - crop_size // 2, crop_size // 2)  # Bottom-left
        # ]
        reference_points = [
            (0, 0),  # Top-left
            (0, captured_width - 0),  # Top-right
            (captured_height - 0, captured_width - 0),  # Bottom-right
            (captured_height - 0, 0)  # Bottom-left
        ]

        # Preprocess the image using dilation and erosion to improve edge detection
        kernel = np.ones((3, 3), np.uint8)  # Define a 3x3 kernel
        img_for_location = cv2.dilate(captured_img, kernel, iterations=2)  # Erosion operation
        img_for_location = cv2.erode(img_for_location, kernel, iterations=2)  # Erosion operation

        # Use Otsu's binarization method for edge detection
        _, edges = cv2.threshold(img_for_location, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = 255 - edges
        # Find the coordinates of all edge points
        edge_points = np.column_stack(np.where(edges > 0))

        # Find the edge point closest to each reference point
        closest_edge_points = []
        for ref_point in reference_points:
            # Calculate the distance from all edge points to the reference point
            distances = np.sqrt((edge_points[:, 0] - ref_point[0]) ** 2 + (edge_points[:, 1] - ref_point[1]) ** 2)
            # Find the index of the smallest distance
            closest_index = np.argmin(distances)
            # Add the closest edge point to the result list
            closest_edge_points.append([edge_points[closest_index][1], edge_points[closest_index][0]])

        # Ensure the order of closest_edge_points matches the template corners
        # Template corners order: top-left, top-right, bottom-right, bottom-left
        template_corners = np.float32([
            [0, 0],
            [template.shape[1] - 1, 0],
            [template.shape[1] - 1, template.shape[0] - 1],
            [0, template.shape[0] - 1]
        ])

        # Perform perspective transformation
        closest_edge_points = np.float32(closest_edge_points)
        M = cv2.getPerspectiveTransform(closest_edge_points, template_corners)
        transformed_image = cv2.warpPerspective(captured_img, M, (template.shape[1], template.shape[0]))
        return transformed_image

    def find_matched_images_by_template(self, captured_img, template):
        """
        Process all images in a folder with perspective transform.

        Parameters:
        - folder_path: Path to the folder containing images
        - src_points_list: List of source points for each image
        - trans_shape: Tuple of (width, height) for the transformed images
        - output_folder: Path to the folder to save transformed images
        """

        result = cv2.matchTemplate(captured_img, template, method=cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        h_y, w_x = template.shape
        top_left = max_loc
        bottom_right = (top_left[0] + w_x, top_left[1] + h_y)
        matched_region = captured_img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        return matched_region


def _unsharp_mask_single_channel(image, radius, amount, vrange):
    """Single channel implementation of the unsharp masking filter."""

    blurred = gaussian_filter(image, sigma=radius, mode='reflect')
    result = image + (image - blurred) * amount
    if vrange is not None:
        return np.clip(result, vrange[0], vrange[1], out=result)
    return result


def unsharp_mask(image, radius=1.0, amount=1.0, multichannel=False, preserve_range=False):
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
            result[..., channel] = _unsharp_mask_single_channel(
                fimg[..., channel], radius, amount, vrange)
        return result
    else:
        return _unsharp_mask_single_channel(fimg, radius, amount, vrange)


def binary(scanned, unsharp_radius=2, unsharp_amount=4):
    """

    :param scanned:
    :param unsharp_radius:
    :param unsharp_amount:
    :return:
    """
    scan = unsharp_mask(scanned, unsharp_radius, unsharp_amount)
    ret, binary_img = cv2.threshold(np.uint8(scan * 255), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_img


def pearson(digital_img, captured_img):
    """
    Calculate the Pearson correlation coefficient between two images using OpenCV's matchTemplate.
    :param digital_img: ndarray of the digital image.
    :param captured_img: ndarray of the captured image.
    :return: Pearson correlation coefficient.
    """
    # Resize the digital image to the size of the captured image
    digital_img_resized = cv2.resize(digital_img, (captured_img.shape[1], captured_img.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
    result = np.corrcoef(captured_img.flatten(), digital_img_resized.flatten())[0, 1]
    return result


def hamming_distance(digital_img, captured_img):
    """
    Calculate the Hamming distance between two images.
    :param digital_img: ndarray of the digital image (already binary).
    :param captured_img: ndarray of the captured image.
    :return: Hamming distance.
    """
    # Resize the digital image to the size of the captured image
    digital_img_resized = cv2.resize(digital_img, (captured_img.shape[1], captured_img.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
    captured_img_binary = binary(captured_img)
    hamming_dist = np.sum(digital_img_resized != captured_img_binary) / digital_img_resized.size
    return hamming_dist


def convert_png_to_pdf_lossless(input_image, output_pdf):
    """

    :param input_image:
    :param output_pdf:
    :return:
    """
    try:
        # Create a canvas object (PDF document)
        c = canvas.Canvas(output_pdf, pagesize=A4)

        # Load the black and white image using ImageReader
        img = ImageReader(input_image)

        # Get the dimensions of the image
        img_width, img_height = img.getSize()

        # Calculate scaling to fit the image onto A4 page
        aspect_ratio = img_width / img_height
        page_width, page_height = A4
        if aspect_ratio >= 1:
            scaled_width = page_width  # Adjust margins as needed
            scaled_height = scaled_width / aspect_ratio
        else:
            scaled_height = page_height
            scaled_width = scaled_height * aspect_ratio

        # Calculate position to center the image on the page
        x_pos = (page_width - scaled_width) / 2
        y_pos = (page_height - scaled_height) / 2

        # Draw the image onto the PDF
        c.drawImage(img, x_pos, y_pos, width=scaled_width, height=scaled_height, preserveAspectRatio=True, mask='auto')

        # Save the PDF document
        c.save()
    finally:
        # Explicitly delete the ImageReader and canvas objects
        del c
