import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

from models.msg import MSG
from models.utils import generate_random_string, find_all_files


def save_datasets(mode_str):
    """
    Display datasets based on the located MSW
    :param mode_str: Mode string indicating the type of dataset to display.
    """
    parser = argparse.ArgumentParser(description="Argument parser for the pearson attack model script.")

    # Common parser arguments
    parser.add_argument('--border_size', type=int, default=12)
    parser.add_argument('--inner_border_mm', type=int, default=5)
    parser.add_argument('--out_border_mm', type=int, default=8)
    parser.add_argument('--outer_border_mm', type=int, default=10)

    # Print arguments
    parser.add_argument('--print_dpi', type=int, default=600)
    parser.add_argument('--scan_ppi', type=int, default=1200)

    # Capture arguments
    parser.add_argument('--load_digital_counterfeit', type=str, default="Images/pearson/attacked_medium")
    parser.add_argument('--load_counterfeit', type=str, default="Images/pearson/counterfeit")
    parser.add_argument('--load_digital', type=str, default="Images/pearson/digital")
    parser.add_argument('--load_genuine', type=str, default="Images/pearson/genuine")
    parser.add_argument('--save_datasets', type=str, default="Images/pearson/datasets")
    args = parser.parse_args()

    load_digital_counterfeit = os.path.join(args.load_digital_counterfeit, mode_str)
    load_counterfeit = os.path.join(args.load_counterfeit, mode_str)

    save_path = os.path.join(args.save_datasets, mode_str)

    os.makedirs(save_path, exist_ok=True)

    path_digital_list = find_all_files(args.load_digital)
    path_digital_list.sort()
    path_genuine_list = find_all_files(args.load_genuine)
    path_genuine_list.sort()
    path_counterfeit_list = find_all_files(load_counterfeit)
    path_counterfeit_list.sort()
    path_digital_counterfeit_list = find_all_files(load_digital_counterfeit)
    path_digital_counterfeit_list.sort()
    img_index = 0
    for digital_path, genuine_path, counterfeit_path, digital_counterfeit_path in tqdm(
            zip(path_digital_list, path_genuine_list, path_counterfeit_list,
                path_digital_counterfeit_list), total=len(path_digital_list),
            desc=f"Process..."):
        img_digital = cv2.imdecode(np.fromfile(f"{digital_path}", dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        img_genuine = cv2.imdecode(np.fromfile(f"{genuine_path}", dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        img_counterfeit = cv2.imdecode(np.fromfile(f"{counterfeit_path}", dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        img_digital_counterfeit = cv2.imdecode(np.fromfile(f"{digital_counterfeit_path}", dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        img_digital_counterfeit_resized = cv2.resize(img_digital_counterfeit, (img_counterfeit.shape[1], img_counterfeit.shape[0]), interpolation=cv2.INTER_NEAREST)
        img_digital_resized = cv2.resize(img_digital, (img_genuine.shape[1], img_genuine.shape[0]), interpolation=cv2.INTER_NEAREST)

        genuine_corr = np.corrcoef(img_digital_resized.flatten(), img_genuine.flatten())[0, 1]
        counterfeit_corr = np.corrcoef(img_digital_resized.flatten(), img_counterfeit.flatten())[0, 1]
        digital_counterfeit_corr = np.corrcoef(img_digital_counterfeit_resized.flatten(), img_counterfeit.flatten())[0, 1]

        if genuine_corr > 0.3 > counterfeit_corr and digital_counterfeit_corr > 0.3:
            cv2.imwrite(f"{save_path}/{img_index:04d}_digital.png", img_digital)
            cv2.imwrite(f"{save_path}/{img_index:04d}_genuine.png", img_genuine)
            cv2.imwrite(f"{save_path}/{img_index:04d}_counterfeit.png", img_counterfeit)
        img_index += 1


if __name__ == "__main__":
    save_datasets("attack_binary")
    save_datasets("attack_network")
    # save_datasets("attack_identity")

