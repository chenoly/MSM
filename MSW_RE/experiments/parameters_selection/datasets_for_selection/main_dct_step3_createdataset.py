import os
import random
import argparse
from tqdm import tqdm
from models.utils import Workflow
from models.mswdcdp import DCT_based_MSG


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # common parser
    parser.add_argument('--data_len', type=int, default=81, help='seed')
    parser.add_argument('--N_list', type=list, default=[12, 24, 36, 48], help='')  # [12, 24, 36, 48] [3, 3, 4, 6]
    parser.add_argument('--alpha_list', type=list, default=[0.5], help='')
    parser.add_argument('--gamma_list', type=list, default=[0.02, 0.04, 0.06, 0.08, 0.1], help='embedding strength')

    parser.add_argument('--border_size', type=int, default=12)
    parser.add_argument('--inner_border_mm', type=int, default=5)
    parser.add_argument('--out_border_mm', type=int, default=8)
    parser.add_argument('--outer_border_mm', type=int, default=10)

    # print
    parser.add_argument('--print_dpi', type=int, default=600)
    parser.add_argument('--scan_ppi', type=int, default=1200)

    # capture
    parser.add_argument('--digital_medium_path', type=str, default=f"Images/dct/digital_match/")
    parser.add_argument('--captured_medium_path', type=str, default=f"Images/dct/scanned_match/")
    parser.add_argument('--save_dir', type=str, default=f"Images/dct/genuine")
    args = parser.parse_args()

    image_index = 0
    WF = Workflow()
    total_iterations = len(args.N_list) * len(args.alpha_list) * len(args.gamma_list)

    # Initialize the tqdm progress bar
    with tqdm(total=total_iterations, desc="Create Datasets") as pbar:
        for N in args.N_list:
            N_path_captured = os.path.join(args.captured_medium_path, f"N_{N}")
            N_path_digital = os.path.join(args.digital_medium_path, f"N_{N}")
            N_path_genuine = os.path.join(args.save_dir, f"N_{N}")
            for alpha in args.alpha_list:
                alpha_path_captured = os.path.join(N_path_captured, f"alpha_{alpha}")
                alpha_path_digital = os.path.join(N_path_digital, f"alpha_{alpha}")
                alpha_path_genuine = os.path.join(N_path_genuine, f"alpha_{alpha}")
                for gamma in args.gamma_list:
                    gamma_path_captured = os.path.join(alpha_path_captured, f"gamma_{gamma}")
                    gamma_path_digital = os.path.join(alpha_path_digital, f"gamma_{gamma}")
                    gamma_path_genuine = os.path.join(alpha_path_genuine, f"gamma_{gamma}")
                    model = DCT_based_MSG(N=N, alpha=alpha, gamma=gamma)
                    marked_code, _ = model.generate_MSG([random.randint(0, 3-1) for _ in range(args.data_len)])
                    WF.createDataset(gamma_path_digital, gamma_path_captured, gamma_path_genuine, mgw_size=marked_code.shape, print_dpi=args.print_dpi, scan_ppi=args.scan_ppi)
                    pbar.update(1)
