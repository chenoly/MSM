import os
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

from models.mswdcdp import DCT_based_MSG
from models.mswpqrcode import PEARSON_based_MSG


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def MSW_P_QR_CODE():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    base_dir = os.path.abspath(os.path.join(current_dir, "..", "datasets_for_selection"))

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

    # capture
    parser.add_argument('--load_digital_dir', type=str, default=f"Images/pearson/digital/")
    parser.add_argument('--load_genuine_dir', type=str, default=f"Images/pearson/genuine/")
    parser.add_argument('--load_netattacked_genuine_dir', type=str, default=f"Images/pearson/netattack_pdf/genuine")
    parser.add_argument('--load_binattacked_genuine_dir', type=str, default=f"Images/pearson/binattack_pdf/genuine")
    args = parser.parse_args()

    total_iterations = len(args.N_list) * len(args.alpha_list) * len(args.gamma_list)

    load_digital_dir = os.path.join(base_dir, args.load_digital_dir)
    load_genuine_dir = os.path.join(base_dir, args.load_genuine_dir)
    load_netattacked_genuine_dir = os.path.join(base_dir, args.load_netattacked_genuine_dir)
    load_binattacked_genuine_dir = os.path.join(base_dir, args.load_binattacked_genuine_dir)

    result = {}

    # Initialize the tqdm progress bar
    with tqdm(total=total_iterations, desc="Create Datasets") as pbar:
        for N in args.N_list:
            result[N] = {}
            N_path_digital = os.path.join(load_digital_dir, f"N_{N}")
            N_path_genuine = os.path.join(load_genuine_dir, f"N_{N}")
            N_path_netattacked = os.path.join(load_netattacked_genuine_dir, f"N_{N}")
            N_path_binattacked = os.path.join(load_binattacked_genuine_dir, f"N_{N}")
            for alpha in args.alpha_list:
                result[N][alpha] = {}
                alpha_path_digital = os.path.join(N_path_digital, f"alpha_{alpha}")
                alpha_path_genuine = os.path.join(N_path_genuine, f"alpha_{alpha}")
                alpha_path_netattacked = os.path.join(N_path_netattacked, f"alpha_{alpha}")
                alpha_path_binattacked = os.path.join(N_path_binattacked, f"alpha_{alpha}")
                for gamma in args.gamma_list:
                    result[N][alpha][gamma] = {"mean_score_gen": [], "mean_score_net": [], "mean_score_bin": [],
                                               "ber_gen": [], "ber_net": [], "ber_bin": [], "neb_gen": [],
                                               "neb_net": [], "neb_bin": [], "gen_corr": [], "net_corr": [],
                                               "bin_corr": []}
                    gamma_path_digital = os.path.join(alpha_path_digital, f"gamma_{gamma}")
                    gamma_path_genuine = os.path.join(alpha_path_genuine, f"gamma_{gamma}")
                    gamma_path_netattacked = os.path.join(alpha_path_netattacked, f"gamma_{gamma}")
                    gamma_path_binattacked = os.path.join(alpha_path_binattacked, f"gamma_{gamma}")
                    digital_paths = sorted([
                        os.path.join(gamma_path_digital, f)
                        for f in os.listdir(gamma_path_digital)
                        if f.lower().endswith(".png")
                    ])

                    genuine_paths = sorted([
                        os.path.join(gamma_path_genuine, f)
                        for f in os.listdir(gamma_path_genuine)
                        if f.lower().endswith(".png")
                    ])

                    netattacked_paths = sorted([
                        os.path.join(gamma_path_netattacked, f)
                        for f in os.listdir(gamma_path_netattacked)
                        if f.lower().endswith(".png")
                    ])

                    binattacked_paths = sorted([
                        os.path.join(gamma_path_binattacked, f)
                        for f in os.listdir(gamma_path_binattacked)
                        if f.lower().endswith(".png")
                    ])
                    if N == 3:
                        theta = 0.02
                    else:
                        theta = 0.01
                    model = PEARSON_based_MSG(N, q=3, alpha=alpha, delta=gamma)
                    model.theta = theta
                    index_save = 0
                    for digital_path, genuine_path, netattacked_path, binattacked_path in zip(digital_paths,
                                                                                              genuine_paths,
                                                                                              netattacked_paths,
                                                                                              binattacked_paths):
                        genuine = np.float32(Image.open(genuine_path).convert('L'))
                        netattacked = np.float32(Image.open(netattacked_path).convert('L'))
                        binattacked = np.float32(Image.open(binattacked_path).convert('L'))
                        digital_resized = np.float32(Image.open(digital_path).convert('L').resize(genuine.shape))
                        digital = np.float32(Image.open(digital_path).convert('L'))
                        file_index = (int(os.path.basename(digital_path).split('.')[0]) - 1)
                        gen_corr = np.corrcoef(digital_resized.flatten(), genuine.flatten())[0, 1]
                        net_corr = np.corrcoef(digital_resized.flatten(), netattacked.flatten())[0, 1]
                        bin_corr = np.corrcoef(digital_resized.flatten(), binattacked.flatten())[0, 1]
                        if gen_corr > 0.4:
                            pattern_list = model.compute_patterns(digital, file_index)
                            emb_bits = model.decode(digital, pattern_list, 1)
                            ext_gen_bits = model.decode(genuine, pattern_list, args.scan_ppi // args.print_dpi)
                            ext_net_bits = model.decode(netattacked, pattern_list, args.scan_ppi // args.print_dpi)
                            ext_bin_bits = model.decode(binattacked, pattern_list, args.scan_ppi // args.print_dpi)

                            mean_score_gen = model.compute_score(genuine, pattern_list, args.scan_ppi // args.print_dpi)
                            mean_score_net = model.compute_score(netattacked, pattern_list,
                                                                 args.scan_ppi // args.print_dpi)
                            mean_score_bin = model.compute_score(binattacked, pattern_list,
                                                                 args.scan_ppi // args.print_dpi)

                            ber_gen = np.mean(np.asarray(emb_bits) != np.asarray(ext_gen_bits))
                            ber_net = np.mean(np.asarray(emb_bits) != np.asarray(ext_net_bits))
                            ber_bin = np.mean(np.asarray(emb_bits) != np.asarray(ext_bin_bits))

                            neb_gen = np.sum(np.asarray(emb_bits) != np.asarray(ext_gen_bits))
                            neb_net = np.sum(np.asarray(emb_bits) != np.asarray(ext_net_bits))
                            neb_bin = np.sum(np.asarray(emb_bits) != np.asarray(ext_bin_bits))

                            result[N][alpha][gamma]["gen_corr"].append(gen_corr)
                            result[N][alpha][gamma]["net_corr"].append(net_corr)
                            result[N][alpha][gamma]["bin_corr"].append(bin_corr)
                            result[N][alpha][gamma]["mean_score_gen"].append(mean_score_gen)
                            result[N][alpha][gamma]["mean_score_net"].append(mean_score_net)
                            result[N][alpha][gamma]["mean_score_bin"].append(mean_score_bin)
                            result[N][alpha][gamma]["ber_gen"].append(ber_gen)
                            result[N][alpha][gamma]["ber_net"].append(ber_net)
                            result[N][alpha][gamma]["ber_bin"].append(ber_bin)

                            result[N][alpha][gamma]["neb_gen"].append(neb_gen)
                            result[N][alpha][gamma]["neb_net"].append(neb_net)
                            result[N][alpha][gamma]["neb_bin"].append(neb_bin)

                            index_save += 1
                            if index_save == 100:
                                break
                            pbar.set_description(
                                f"N: {N}, alpha: {alpha}, gamma: {gamma}, score_gen: {mean_score_gen:.4f}"
                                f", score_net: {mean_score_net:.4f}, score_bin: {mean_score_bin:.4f}, "
                                f"ber_gen: {ber_gen:.4f}, ber_net: {ber_net:.4f}, ber_bin: {ber_bin:.4f}, "
                                f"neb_gen: {neb_gen:.4f}, neb_net: {neb_net:.4f}, neb_bin: {neb_bin:.4f}, "
                                f"index: {index_save}")
                    pbar.update(1)

                    os.makedirs("save_metrics", exist_ok=True)
                    with open("save_metrics/pearson_metrics.json", "w") as f:
                        json.dump(result, f, cls=NumpyEncoder, indent=4)


def MSW_D_CDP():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_len', type=int, default=81)
    parser.add_argument('--N_list', type=list, default=[12, 24, 36, 48], help='')
    parser.add_argument('--alpha_list', type=list, default=[0.5], help='')
    parser.add_argument('--gamma_list', type=list, default=[0.02, 0.04, 0.06, 0.08, 0.1])
    parser.add_argument('--border_size', type=int, default=12)
    parser.add_argument('--inner_border_mm', type=int, default=5)
    parser.add_argument('--out_border_mm', type=int, default=8)
    parser.add_argument('--outer_border_mm', type=int, default=10)

    # print
    parser.add_argument('--print_dpi', type=int, default=600)
    parser.add_argument('--scan_ppi', type=int, default=1200)

    # capture
    parser.add_argument('--load_digital_dir', type=str, default=f"Images/dct/digital/")
    parser.add_argument('--load_genuine_dir', type=str, default=f"Images/dct/genuine/")
    parser.add_argument('--load_netattacked_genuine_dir', type=str, default=f"Images/dct/netattack_pdf/genuine")
    parser.add_argument('--load_binattacked_genuine_dir', type=str, default=f"Images/dct/binattack_pdf/genuine")
    args = parser.parse_args()

    total_iterations = len(args.N_list) * len(args.alpha_list) * len(args.gamma_list)

    current_dir = os.path.dirname(os.path.abspath(__file__))

    base_dir = os.path.abspath(os.path.join(current_dir, "..", "datasets_for_selection"))


    load_digital_dir = os.path.join(base_dir, args.load_digital_dir)
    load_genuine_dir = os.path.join(base_dir, args.load_genuine_dir)
    load_netattacked_genuine_dir = os.path.join(base_dir, args.load_netattacked_genuine_dir)
    load_binattacked_genuine_dir = os.path.join(base_dir, args.load_binattacked_genuine_dir)

    result = {}

    # Initialize the tqdm progress bar
    with tqdm(total=total_iterations, desc="Create Datasets") as pbar:
        for N in args.N_list:
            result[N] = {}
            N_path_digital = os.path.join(load_digital_dir, f"N_{N}")
            N_path_genuine = os.path.join(load_genuine_dir, f"N_{N}")
            N_path_netattacked = os.path.join(load_netattacked_genuine_dir, f"N_{N}")
            N_path_binattacked = os.path.join(load_binattacked_genuine_dir, f"N_{N}")
            for alpha in args.alpha_list:
                result[N][alpha] = {}
                alpha_path_digital = os.path.join(N_path_digital, f"alpha_{alpha}")
                alpha_path_genuine = os.path.join(N_path_genuine, f"alpha_{alpha}")
                alpha_path_netattacked = os.path.join(N_path_netattacked, f"alpha_{alpha}")
                alpha_path_binattacked = os.path.join(N_path_binattacked, f"alpha_{alpha}")
                for gamma in args.gamma_list:
                    result[N][alpha][gamma] = {"mean_score_gen": [], "mean_score_net": [], "mean_score_bin": [],
                                               "ber_gen": [], "ber_net": [], "ber_bin": [], "neb_gen": [],
                                               "neb_net": [], "neb_bin": [], "gen_corr": [], "net_corr": [],
                                               "bin_corr": []}
                    gamma_path_digital = os.path.join(alpha_path_digital, f"gamma_{gamma}")
                    gamma_path_genuine = os.path.join(alpha_path_genuine, f"gamma_{gamma}")
                    gamma_path_netattacked = os.path.join(alpha_path_netattacked, f"gamma_{gamma}")
                    gamma_path_binattacked = os.path.join(alpha_path_binattacked, f"gamma_{gamma}")
                    digital_paths = sorted([
                        os.path.join(gamma_path_digital, f)
                        for f in os.listdir(gamma_path_digital)
                        if f.lower().endswith(".png")
                    ])

                    genuine_paths = sorted([
                        os.path.join(gamma_path_genuine, f)
                        for f in os.listdir(gamma_path_genuine)
                        if f.lower().endswith(".png")
                    ])

                    netattacked_paths = sorted([
                        os.path.join(gamma_path_netattacked, f)
                        for f in os.listdir(gamma_path_netattacked)
                        if f.lower().endswith(".png")
                    ])

                    binattacked_paths = sorted([
                        os.path.join(gamma_path_binattacked, f)
                        for f in os.listdir(gamma_path_binattacked)
                        if f.lower().endswith(".png")
                    ])
                    model = DCT_based_MSG(N=N, alpha=alpha, gamma=gamma)
                    index_save = 0
                    for digital_path, genuine_path, netattacked_path, binattacked_path in zip(digital_paths,
                                                                                              genuine_paths,
                                                                                              netattacked_paths,
                                                                                              binattacked_paths):
                        genuine = np.float32(Image.open(genuine_path).convert('L'))
                        netattacked = np.float32(Image.open(netattacked_path).convert('L'))
                        binattacked = np.float32(Image.open(binattacked_path).convert('L'))
                        digital_resized = np.float32(Image.open(digital_path).convert('L').resize(genuine.shape))
                        digital = np.float32(Image.open(digital_path).convert('L'))
                        file_index = (int(os.path.basename(digital_path).split('.')[0]) - 1)
                        gen_corr = np.corrcoef(digital_resized.flatten(), genuine.flatten())[0, 1]
                        net_corr = np.corrcoef(digital_resized.flatten(), netattacked.flatten())[0, 1]
                        bin_corr = np.corrcoef(digital_resized.flatten(), binattacked.flatten())[0, 1]
                        if gen_corr > 0.4:
                            pattern_list = model.compute_patterns(digital, file_index)
                            emb_bits = model.decode(digital)
                            ext_gen_bits = model.decode(np.float32(Image.fromarray(np.uint8(genuine)).resize(digital.shape)))
                            ext_net_bits = model.decode(np.float32(Image.fromarray(np.uint8(netattacked)).resize(digital.shape)))
                            ext_bin_bits = model.decode(np.float32(Image.fromarray(np.uint8(binattacked)).resize(digital.shape)))

                            mean_score_gen = model.compute_score(genuine, pattern_list, args.scan_ppi // args.print_dpi)
                            mean_score_net = model.compute_score(netattacked, pattern_list, args.scan_ppi // args.print_dpi)
                            mean_score_bin = model.compute_score(binattacked, pattern_list, args.scan_ppi // args.print_dpi)

                            ber_gen = np.mean(np.asarray(emb_bits) != np.asarray(ext_gen_bits))
                            ber_net = np.mean(np.asarray(emb_bits) != np.asarray(ext_net_bits))
                            ber_bin = np.mean(np.asarray(emb_bits) != np.asarray(ext_bin_bits))

                            neb_gen = np.sum(np.asarray(emb_bits) != np.asarray(ext_gen_bits))
                            neb_net = np.sum(np.asarray(emb_bits) != np.asarray(ext_net_bits))
                            neb_bin = np.sum(np.asarray(emb_bits) != np.asarray(ext_bin_bits))

                            result[N][alpha][gamma]["gen_corr"].append(gen_corr)
                            result[N][alpha][gamma]["net_corr"].append(net_corr)
                            result[N][alpha][gamma]["bin_corr"].append(bin_corr)
                            result[N][alpha][gamma]["mean_score_gen"].append(mean_score_gen)
                            result[N][alpha][gamma]["mean_score_net"].append(mean_score_net)
                            result[N][alpha][gamma]["mean_score_bin"].append(mean_score_bin)
                            result[N][alpha][gamma]["ber_gen"].append(ber_gen)
                            result[N][alpha][gamma]["ber_net"].append(ber_net)
                            result[N][alpha][gamma]["ber_bin"].append(ber_bin)

                            result[N][alpha][gamma]["neb_gen"].append(neb_gen)
                            result[N][alpha][gamma]["neb_net"].append(neb_net)
                            result[N][alpha][gamma]["neb_bin"].append(neb_bin)

                            index_save += 1
                            if index_save == 100:
                                break

                            pbar.set_description(
                                f"N: {N}, alpha: {alpha}, gamma: {gamma}, score_gen: {mean_score_gen:.4f}"
                                f", score_net: {mean_score_net:.4f}, score_bin: {mean_score_bin:.4f}, "
                                f"ber_gen: {ber_gen:.4f}, ber_net: {ber_net:.4f}, ber_bin: {ber_bin:.4f}, "
                                f"neb_gen: {neb_gen:.4f}, neb_net: {neb_net:.4f}, neb_bin: {neb_bin:.4f}, "
                                f"index: {index_save}")
                    pbar.update(1)

                    os.makedirs("save_metrics", exist_ok=True)
                    with open("save_metrics/dct_metrics.json", "w") as f:
                        json.dump(result, f, cls=NumpyEncoder, indent=4)


if __name__ == "__main__":
    MSW_D_CDP()
    # MSW_P_QR_CODE()
