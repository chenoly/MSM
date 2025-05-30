import argparse
import glob
import os.path
import shutil

from models.utils import Workflow
from models.mswpqrcode import PEARSON_based_MSG

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # common parser
    parser.add_argument('--data_len', type=int, default=81, help='seed')

    parser.add_argument('--N_list', type=list, default=[12, 24, 36, 48], help='')  # [12, 24, 36, 48] [3, 3, 4, 6]
    parser.add_argument('--alpha_list', type=list, default=[0.5], help='')
    parser.add_argument('--gamma_list', type=list, default=[0.02, 0.04, 0.06, 0.08, 0.1], help='embedding strength')


    parser.add_argument('--border_size', type=int, default=12, help='val_epoch')
    parser.add_argument('--inner_border_mm', type=int, default=5, help='val_epoch')
    parser.add_argument('--out_border_mm', type=int, default=8, help='val_epoch')
    parser.add_argument('--outer_border_mm', type=int, default=10, help='val_epoch')

    # print
    parser.add_argument('--Nums', type=list, default=[2, 3, 4, 6], help='seed')
    parser.add_argument('--print_dpi', type=int, default=600, help='Print resolution in dots per inch (DPI).')
    parser.add_argument('--load_digital_located_dir', type=str, default="Images/dct/digital_located")
    parser.add_argument('--load_scanned_located_dir', type=str, default="Images/dct/scanned_located")
    parser.add_argument('--save_digital_match_dir', type=str, default="Images/dct/digital_match")
    parser.add_argument('--save_scanned_match_dir', type=str, default="Images/dct/scanned_match")
    args = parser.parse_args()

    WF = Workflow()
    for N, Num in zip(args.N_list, args.Nums):
        for alpha in args.alpha_list:
            for gamma in args.gamma_list:
                matched_files = glob.glob(
                    os.path.join(args.load_digital_located_dir, f"dpi_{args.print_dpi}_*_{N}_{alpha}_{gamma}.png"))
                save_path = os.path.join(args.save_digital_match_dir, f"N_{N}", f"alpha_{alpha}", f"gamma_{gamma}")
                os.makedirs(save_path, exist_ok=True)
                for src_file in matched_files:
                    filename = os.path.basename(src_file)
                    dst_file = os.path.join(save_path, filename)
                    shutil.copy2(src_file, dst_file)

                matched_files = glob.glob(
                    os.path.join(args.load_scanned_located_dir, f"dpi_{args.print_dpi}_*_{N}_{alpha}_{gamma}.png"))
                save_path = os.path.join(args.save_scanned_match_dir, f"N_{N}", f"alpha_{alpha}", f"gamma_{gamma}")
                os.makedirs(save_path, exist_ok=True)
                for src_file in matched_files:
                    filename = os.path.basename(src_file)
                    dst_file = os.path.join(save_path, filename)
                    shutil.copy2(src_file, dst_file)
