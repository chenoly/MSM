import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from models.attacker import NetAttack, BinAttack

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_len', type=int, default=81)
    parser.add_argument('--N_list', type=list, default=[3, 6, 9, 12])
    parser.add_argument('--alpha_list', type=list, default=[0.583])
    parser.add_argument('--gamma_list', type=list, default=[-0.2, -0.1, 0.0, 0.1, 0.2])
    parser.add_argument('--border_size', type=int, default=12)
    parser.add_argument('--inner_border_mm', type=int, default=5)
    parser.add_argument('--out_border_mm', type=int, default=8)
    parser.add_argument('--outer_border_mm', type=int, default=10)

    # print
    parser.add_argument('--print_dpi', type=int, default=600)
    parser.add_argument('--scan_ppi', type=int, default=1200)

    # model path
    parser.add_argument('--model_path', type=str, default=f"E:\研究生\主要研究工作\物理不可复制防伪水印\源代码\MSW_NEW\experiments/attack/train_attack_model/checkpoints/generator_0.6.pth")

    # capture
    parser.add_argument('--genuine_dir', type=str, default=f"Images/pearson/genuine/")
    parser.add_argument('--save_netattacked_dir', type=str, default=f"Images/pearson/counterfeit/NetAttack")
    parser.add_argument('--save_binattacked_dir', type=str, default=f"Images/pearson/counterfeit/BinAttack")
    args = parser.parse_args()

    netattack = NetAttack(args.model_path, args.print_dpi, args.scan_ppi)

    binattack = BinAttack(args.print_dpi, args.scan_ppi)

    total_iterations = len(args.N_list) * len(args.alpha_list) * len(args.gamma_list)

    # Initialize the tqdm progress bar
    with tqdm(total=total_iterations, desc="Create Datasets") as pbar:
        for N in args.N_list:
            N_path_genuine = os.path.join(args.genuine_dir, f"N_{N}")
            N_path_netattacked = os.path.join(args.save_netattacked_dir, f"N_{N}")
            N_path_binattacked = os.path.join(args.save_binattacked_dir, f"N_{N}")
            for alpha in args.alpha_list:
                alpha_path_genuine = os.path.join(N_path_genuine, f"alpha_{alpha}")
                alpha_path_netattacked = os.path.join(N_path_netattacked, f"alpha_{alpha}")
                alpha_path_binattacked = os.path.join(N_path_binattacked, f"alpha_{alpha}")
                for gamma in args.gamma_list:
                    gamma_path_genuine = os.path.join(alpha_path_genuine, f"gamma_{gamma}")
                    gamma_path_netattacked = os.path.join(alpha_path_netattacked, f"gamma_{gamma}")
                    gamma_path_binattacked = os.path.join(alpha_path_binattacked, f"gamma_{gamma}")
                    os.makedirs(gamma_path_netattacked, exist_ok=True)
                    os.makedirs(gamma_path_binattacked, exist_ok=True)
                    image_files = [f for f in os.listdir(gamma_path_genuine) if f.lower().endswith((".png", ".jpg", ".bmp", ".jpeg"))]
                    for filename in image_files:
                        input_path = os.path.join(gamma_path_genuine, filename)
                        netattack_output_path = os.path.join(gamma_path_netattacked, filename)
                        binattack_output_path = os.path.join(gamma_path_binattacked, filename)
                        genuine_code = Image.open(input_path).convert('L')
                        netattacked_code = netattack(np.float32(genuine_code))
                        binattacked_code = binattack(np.float32(genuine_code), unsharp_radius=6, unsharp_amount=5)
                        Image.fromarray(netattacked_code).save(netattack_output_path)
                        Image.fromarray(binattacked_code).save(binattack_output_path)
                    pbar.update(1)


