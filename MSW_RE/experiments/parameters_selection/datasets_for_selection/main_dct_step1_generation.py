import os
import argparse
from models.utils import Workflow
from models.mswdcdp import DCT_based_MSG

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # common parser
    parser.add_argument('--data_len', type=int, default=81, help='seed')

    parser.add_argument('--N_list', type=list, default=[12, 24, 36, 48], help='')  # [12, 24, 36, 48] [3, 3, 4, 6]
    parser.add_argument('--alpha_list', type=list, default=[0.5], help='')
    parser.add_argument('--gamma_list', type=list, default=[0.02, 0.04, 0.06, 0.08, 0.1], help='embedding strength')

    parser.add_argument('--mode', type=str, default='dct', help='val_epoch')  # pearson

    parser.add_argument('--border_size', type=int, default=12, help='val_epoch')
    parser.add_argument('--inner_border_mm', type=int, default=5, help='val_epoch')
    parser.add_argument('--out_border_mm', type=int, default=8, help='val_epoch')
    parser.add_argument('--outer_border_mm', type=int, default=10, help='val_epoch')

    # print
    parser.add_argument('--Nums', type=list, default=[2, 3, 4, 6], help='seed')
    parser.add_argument('--print_dpi', type=int, default=600, help='val_epoch')
    parser.add_argument('--save_dir', type=str, default=f"Images/dct/", help='val_epoch')
    parser.add_argument('--save_scan_dir', type=str, default="Images/dct/scan_captured")
    args = parser.parse_args()

    WF = Workflow()
    for N, Num in zip(args.N_list, args.Nums):
        save_N_path = os.path.join(args.save_scan_dir, f"N_{N}")
        for alpha in args.alpha_list:
            save_alpha_path = os.path.join(save_N_path, f"alpha_{alpha}")
            for gamma in args.gamma_list:
                save_gamma_path = os.path.join(save_alpha_path, f"gamma_{gamma}")
                os.makedirs(save_gamma_path, exist_ok=True)
                model = DCT_based_MSG(N=N, alpha=alpha, gamma=gamma)
                WF.generate(args.save_dir, Num, model, data_len=args.data_len, alpha=alpha, N=N, gamma=gamma, print_dpi=args.print_dpi, inner_border_mm=args.inner_border_mm,
                            out_border_mm=args.out_border_mm, outer_border_mm=args.outer_border_mm)
    WF.merge_and_delete_pdfs(f"{args.save_dir}/digital_outer/")
