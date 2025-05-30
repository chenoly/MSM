import argparse
import os.path
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
    parser.add_argument('--save_dir', type=str, default="Images/pearson/", help='Directory to save the images.')
    parser.add_argument('--save_scan_dir', type=str, default="Images/pearson/scan_captured", help='Directory to save the images.')
    args = parser.parse_args()

    WF = Workflow()
    for N, Num in zip(args.N_list, args.Nums):
        for alpha in args.alpha_list:
            for gamma in args.gamma_list:
                if N == 3:
                    theta = 0.02
                else:
                    theta = 0.01
                model = PEARSON_based_MSG(N, q=3, alpha=alpha, delta=gamma)
                model.theta = theta
                WF.generate(args.save_dir, Num, model, data_len=args.data_len, N=N, alpha=alpha, gamma=gamma, print_dpi=args.print_dpi, inner_border_mm=args.inner_border_mm, out_border_mm=args.out_border_mm, outer_border_mm=args.outer_border_mm)
    WF.merge_and_delete_pdfs(f"{args.save_dir}/digital_outer/")