import os
import shutil

import cv2
import argparse
import numpy as np
from tqdm import tqdm
from models.utils import Workflow, find_all_files, Attacker

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Argument parser for the attack model script.")

    # Border parameters
    parser.add_argument('--inner_border_mm', type=int, default=5, help='Inner border size in millimeters.')
    parser.add_argument('--out_border_mm', type=int, default=8, help='Outer border size in millimeters.')
    parser.add_argument('--outer_border_mm', type=int, default=10, help='Outer border size in millimeters.')

    # Scan and print parameters
    parser.add_argument('--Num', type=int, default=10, help='Training data size.')
    parser.add_argument('--scan_ppi', type=int, default=1200, help='Scan resolution in pixels per inch (PPI).')
    parser.add_argument('--print_dpi', type=int, default=600, help='Print resolution in dots per inch (DPI).')

    # Training parameters
    parser.add_argument('--train_size', type=int, default=64, help='Training data size.')

    # Model paths
    parser.add_argument('--model_pth_path', type=str, default=r"/attack/train_attack_model/checkpoints/encoder_net_alpha_0_5.pth", help='Paths to the model checkpoint files.')

    # File paths
    parser.add_argument('--load_genuine', type=str, default=r"Images/dct/genuine", help='Path to load the captured medium images.')
    parser.add_argument('--save_attack_medium', type=str, default=r"Images/dct/attacked_medium", help='Path to save the attacked medium images.')
    parser.add_argument('--save_attack_digital', type=str, default=r"Images/dct/attacked_medium/attack_digital", help='Path to save the attacked medium images.')
    parser.add_argument('--save_attack_captured', type=str, default=r"Images/dct/attack_captured_all", help='Path to save the attacked medium images.')

    args = parser.parse_args()

    base_path = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
    model_path_path = os.path.join(base_path, args.model_pth_path.lstrip("/\\"))

    img_index = 0
    WF = Workflow()
    attacker = Attacker(args.print_dpi, args.scan_ppi)

    all_image_list = find_all_files(args.load_genuine)
    all_image_list.sort()

    os.makedirs(args.save_attack_medium, exist_ok=True)
    os.makedirs(args.save_attack_digital, exist_ok=True)
    for idx, img_dir in enumerate(tqdm(all_image_list, desc=f"Processing Images:", leave=False)):
        captured_img = cv2.imdecode(np.fromfile(img_dir, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        nw_attack = attacker.attack(captured_img, method="network", model_path=model_path_path, window_size=args.train_size)
        no_attack = attacker.attack(captured_img, method="identity")
        bn_attack = attacker.attack(captured_img, method="binary")

        cv2.imwrite(os.path.join(args.save_attack_digital, f"{idx + 1:04d}_identity.png"), no_attack)
        cv2.imwrite(os.path.join(args.save_attack_digital, f"{idx + 1:04d}_binary.png"), bn_attack)
        cv2.imwrite(os.path.join(args.save_attack_digital, f"{idx + 1:04d}_network.png"), nw_attack)

    WF.generate_for_attack(args.Num, args.save_attack_medium, args.save_attack_captured, args.save_attack_digital, args.print_dpi, args.inner_border_mm, args.out_border_mm, args.outer_border_mm)

    WF.merge_and_delete_pdfs(f"{args.save_attack_medium}/attack_outer/attack_identity/")
    WF.merge_and_delete_pdfs(f"{args.save_attack_medium}/attack_outer/attack_binary/")
    WF.merge_and_delete_pdfs(f"{args.save_attack_medium}/attack_outer/attack_network/")

    load_binary_located_path_list = find_all_files(args.save_attack_digital)
    load_binary_located_path_list.sort()
    with tqdm(total=len(load_binary_located_path_list) // 3, desc="Processing Images") as pbar:
        for index in range(0, len(load_binary_located_path_list), 3):
            attack_identity_path = os.path.join(args.save_attack_medium, "attack_identity")
            attack_binary_path = os.path.join(args.save_attack_medium, "attack_binary")
            attack_network_path = os.path.join(args.save_attack_medium, "attack_network")
            os.makedirs(os.path.join(args.save_attack_medium, "attack_identity"), exist_ok=True)
            os.makedirs(os.path.join(args.save_attack_medium, "attack_binary"), exist_ok=True)
            os.makedirs(os.path.join(args.save_attack_medium, "attack_network"), exist_ok=True)
            shutil.copy(load_binary_located_path_list[index], attack_identity_path)
            shutil.copy(load_binary_located_path_list[index + 1], attack_binary_path)
            shutil.copy(load_binary_located_path_list[index + 2], attack_network_path)
            pbar.update(1)