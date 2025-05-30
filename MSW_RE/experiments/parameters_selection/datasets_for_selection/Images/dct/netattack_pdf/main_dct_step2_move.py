import glob
import itertools
import shutil
import os.path
import argparse

from tqdm import tqdm

from models.utils import Workflow

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Argument parser for the script.")

    # Common parser arguments
    parser.add_argument('--data_len', type=int, default=81)
    parser.add_argument('--N_list', type=list, default=[12, 24, 36, 48], help='')  # [12, 24, 36, 48] [3, 3, 4, 6]
    parser.add_argument('--alpha_list', type=list, default=[0.5], help='')
    parser.add_argument('--gamma_list', type=list, default=[0.02, 0.04, 0.06, 0.08, 0.1], help='embedding strength')

    parser.add_argument('--border_size', type=int, default=12)
    parser.add_argument('--inner_border_mm', type=int, default=5)
    parser.add_argument('--out_border_mm', type=int, default=8)
    parser.add_argument('--outer_border_mm', type=int, default=10)

    # Print arguments
    parser.add_argument('--Nums', type=list, default=[2, 3, 4, 6], help='seed')
    parser.add_argument('--print_dpi', type=int, default=600)
    parser.add_argument('--load_digital_located_dir', type=str, default="digital_located/")
    parser.add_argument('--load_scanned_located_dir', type=str, default="scanned_located/")
    parser.add_argument('--save_digital_match_dir', type=str, default="digital_match/")
    parser.add_argument('--save_scanned_match_dir', type=str, default="scanned_match/")
    args = parser.parse_args()

    all_combinations = list(itertools.product(args.N_list, args.alpha_list, args.gamma_list))
    with tqdm(total=len(all_combinations), desc="Processing All Combinations", unit="combo") as pbar:
        for N, alpha, gamma in all_combinations:
            pbar.set_description(f"Processing N={N}, alpha={alpha}, gamma={gamma}")
            matched_files = glob.glob(os.path.join(args.load_digital_located_dir, f"dpi_{args.print_dpi}_*_{N}_{alpha}_{gamma}.png"))
            save_path = os.path.join(args.save_digital_match_dir, f"N_{N}", f"alpha_{alpha}", f"gamma_{gamma}")
            os.makedirs(save_path, exist_ok=True)
            for src_file in matched_files:
                filename = os.path.basename(src_file)
                dst_file = os.path.join(save_path, filename)
                shutil.move(src_file, dst_file)

            matched_files = glob.glob(
                os.path.join(args.load_scanned_located_dir, f"d_netattack_dpi_{args.print_dpi}_*_{N}_{alpha}_{gamma}.png"))
            save_path = os.path.join(args.save_scanned_match_dir, f"N_{N}", f"alpha_{alpha}", f"gamma_{gamma}")
            os.makedirs(save_path, exist_ok=True)
            for src_file in matched_files:
                filename = os.path.basename(src_file)
                dst_file = os.path.join(save_path, filename)
                shutil.move(src_file, dst_file)
            pbar.update(1)
