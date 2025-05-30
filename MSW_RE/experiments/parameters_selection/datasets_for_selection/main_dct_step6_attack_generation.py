import os
import argparse
from tqdm import tqdm
from models.attacked_model import AttackModel
from models.attacker import NetAttack, BinAttack
from models.utils import Workflow

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
    parser.add_argument('--Nums', type=list, default=[2, 3, 4, 6], help='seed')
    parser.add_argument('--genuine_dir', type=str, default=f"Images/dct/genuine/")
    parser.add_argument('--save_netattack_dir', type=str, default=f"Images/dct/netattack_pdf/")
    parser.add_argument('--save_binattack_dir', type=str, default=f"Images/dct/binattack_pdf/")
    parser.add_argument('--netattacked_dir', type=str, default=f"Images/dct/counterfeit/NetAttack")
    parser.add_argument('--binattacked_dir', type=str, default=f"Images/dct/counterfeit/BinAttack")
    args = parser.parse_args()

    WF = Workflow()

    image_index = 0
    total_iterations = len(args.N_list) * len(args.alpha_list) * len(args.gamma_list)

    # Initialize the tqdm progress bar
    with tqdm(total=total_iterations, desc="Create Datasets") as pbar:
        for N, Num in zip(args.N_list, args.Nums):
            N_path_genuine = os.path.join(args.genuine_dir, f"N_{N}")
            N_path_netattacked = os.path.join(args.netattacked_dir, f"N_{N}")
            N_path_binattacked = os.path.join(args.binattacked_dir, f"N_{N}")
            for alpha in args.alpha_list:
                alpha_path_genuine = os.path.join(N_path_genuine, f"alpha_{alpha}")
                alpha_path_netattacked = os.path.join(N_path_netattacked, f"alpha_{alpha}")
                alpha_path_binattacked = os.path.join(N_path_binattacked, f"alpha_{alpha}")
                for gamma in args.gamma_list:
                    gamma_path_genuine = os.path.join(alpha_path_genuine, f"gamma_{gamma}")
                    gamma_path_netattacked = os.path.join(alpha_path_netattacked, f"gamma_{gamma}")
                    gamma_path_binattacked = os.path.join(alpha_path_binattacked, f"gamma_{gamma}")

                    netattack_model = AttackModel(gamma_path_netattacked, N, alpha, gamma, q=2)
                    WF.generate(args.save_netattack_dir, Num, netattack_model, data_len=args.data_len, N=N, alpha=alpha,
                                gamma=gamma, print_dpi=args.print_dpi, inner_border_mm=args.inner_border_mm,
                                out_border_mm=args.out_border_mm, outer_border_mm=args.outer_border_mm)

                    binattack_model = AttackModel(gamma_path_binattacked, N, alpha, gamma, q=2)
                    WF.generate(args.save_binattack_dir, Num, binattack_model, data_len=args.data_len, N=N, alpha=alpha,
                                gamma=gamma, print_dpi=args.print_dpi, inner_border_mm=args.inner_border_mm,
                                out_border_mm=args.out_border_mm, outer_border_mm=args.outer_border_mm)
    WF.merge_and_delete_pdfs(f"{args.save_netattack_dir}/digital_outer/")
    WF.merge_and_delete_pdfs(f"{args.save_binattack_dir}/digital_outer/")
