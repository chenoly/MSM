import argparse
import glob
import os.path
import shutil

from models.utils import Workflow
from models.mswpqrcode import PEARSON_based_MSG

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Argument parser for the script.")

    # Common parser arguments
    parser.add_argument('--data_len', type=int, default=81, help='Length of the data.')
    parser.add_argument('--N_list', type=list, default=[3, 6, 9, 12], help='List of N values.')
    parser.add_argument('--alpha_list', type=list, default=[0.583], help='List of alpha values.')
    parser.add_argument('--gamma_list', type=list, default=[-0.2, -0.1, 0.0, 0.1, 0.2], help='List of gamma values representing embedding strength.')

    parser.add_argument('--border_size', type=int, default=12, help='Size of the border in pixels.')
    parser.add_argument('--inner_border_mm', type=int, default=5, help='Inner border size in millimeters.')
    parser.add_argument('--out_border_mm', type=int, default=8, help='Outer border size in millimeters.')
    parser.add_argument('--outer_border_mm', type=int, default=10, help='Outer border size in millimeters.')

    # Print arguments
    parser.add_argument('--Nums', type=list, default=[1, 1, 2, 2], help='seed')
    parser.add_argument('--print_dpi', type=int, default=600, help='Print resolution in dots per inch (DPI).')
    parser.add_argument('--load_digital_located_dir', type=str, default="Images/pearson/digital_located")
    parser.add_argument('--load_scanned_located_dir', type=str, default="Images/pearson/scanned_located")
    parser.add_argument('--save_digital_match_dir', type=str, default="Images/pearson/digital_match")
    parser.add_argument('--save_scanned_match_dir', type=str, default="Images/pearson/scanned_match")
    args = parser.parse_args()

    WF = Workflow()
    for N, Num in zip(args.N_list, args.Nums):
        for alpha in args.alpha_list:
            for gamma in args.gamma_list:
                if gamma == 0:
                    gamma_str = "*0.0"
                else:
                    gamma_str = gamma
                matched_files = glob.glob(os.path.join(args.load_digital_located_dir, f"dpi_{args.print_dpi}_*_{N}_{alpha}_{gamma_str}.png"))
                save_path = os.path.join(args.save_digital_match_dir, f"N_{N}", f"alpha_{alpha}", f"gamma_{gamma}")
                os.makedirs(save_path, exist_ok=True)
                for src_file in matched_files:
                    filename = os.path.basename(src_file)
                    dst_file = os.path.join(save_path, filename)
                    shutil.copy2(src_file, dst_file)

                if gamma == 0:
                    gamma_str = "*0.0"
                else:
                    gamma_str = gamma
                matched_files = glob.glob(os.path.join(args.load_scanned_located_dir, f"dpi_{args.print_dpi}_*_{N}_{alpha}_{gamma_str}.png"))
                save_path = os.path.join(args.save_scanned_match_dir, f"N_{N}", f"alpha_{alpha}", f"gamma_{gamma}")
                os.makedirs(save_path, exist_ok=True)
                for src_file in matched_files:
                    filename = os.path.basename(src_file)
                    dst_file = os.path.join(save_path, filename)
                    shutil.copy2(src_file, dst_file)