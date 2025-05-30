import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from models.attacker import NetAttack, BinAttack

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--border_size', type=int, default=12)
    parser.add_argument('--inner_border_mm', type=int, default=5)
    parser.add_argument('--out_border_mm', type=int, default=8)
    parser.add_argument('--outer_border_mm', type=int, default=10)

    # print
    parser.add_argument('--print_dpi', type=int, default=600)
    parser.add_argument('--scan_ppi', type=int, default=1200)

    # model path
    parser.add_argument('--model_path', type=str,
                        default=f"E:\研究生\主要研究工作\物理不可复制防伪水印\源代码\MSW_NEW\experiments/attack/train_attack_model/checkpoints/generator_0.5.pth")

    # capture
    parser.add_argument('--method_list', type=str, default=["2lqrcode", "cdp", "dct", "pearson"])
    parser.add_argument('--base_dir', type=str, default=f"Images/")
    args = parser.parse_args()

    binattack = BinAttack(args.print_dpi, args.scan_ppi)

    # Initialize the tqdm progress bar
    for method_str in args.method_list:

        if method_str in ["pearson", "2lqrcode"]:
            model_path = args.model_path.replace('0.5', '0.6')
        else:
            model_path = args.model_path

        print(f"load: {model_path}")

        netattack = NetAttack(model_path, args.print_dpi, args.scan_ppi)

        netattack_dir = os.path.join(args.base_dir, method_str, "attacked", "netattack")
        os.makedirs(netattack_dir, exist_ok=True)
        binattack_dir = os.path.join(args.base_dir, method_str, "attacked", "binattack")
        os.makedirs(binattack_dir, exist_ok=True)

        genuine_dir = os.path.join(args.base_dir, method_str, "genuine")
        image_files = [os.path.join(genuine_dir, f) for f in os.listdir(genuine_dir) if f.lower().endswith('.png')]
        for image_file in image_files:
            print(f"file name:{image_file}")
            filename = os.path.basename(image_file)
            genuine_code = Image.open(image_file).convert('L')
            netattacked_code = netattack(np.float32(genuine_code))
            binattacked_code = binattack(np.float32(genuine_code))
            Image.fromarray(netattacked_code).save(os.path.join(netattack_dir, filename))
            Image.fromarray(binattacked_code).save(os.path.join(binattack_dir, filename))
