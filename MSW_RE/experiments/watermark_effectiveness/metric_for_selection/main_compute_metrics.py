import os
import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm
from models import utils
from models.utils import find_all_files


def group_elements(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def compute_cdp_dct_metric(args, mode_str):
    """
    Computes metrics for MSW.

    :param args: Arguments containing lists of N, alpha, and gamma values.
    :param mode_str: Mode of the computation ('pearson' or 'dct').
    :return: Dictionary containing computed metrics.
    """
    metric_dict = {}
    v = args.scan_ppi // args.print_dpi
    base_path = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
    dct_path = os.path.join(base_path, args.load_dct_path.lstrip("/\\"))
    load_dct_path = os.path.join(dct_path, mode_str)
    load_cdp_path = os.path.join(base_path, args.load_cdp_path.lstrip("/\\"))
    load_cdp_path = os.path.join(load_cdp_path, mode_str)

    all_dct_image_list = find_all_files(load_dct_path)
    all_dct_image_list.sort()
    all_dct_image_list = list(group_elements(all_dct_image_list, 3))
    all_cdp_image_list = find_all_files(load_cdp_path)
    all_cdp_image_list.sort()
    all_cdp_image_list = list(group_elements(all_cdp_image_list, 3))
    dct_g_result_list = []
    dct_c_result_list = []
    cdp_g_result_list = []
    cdp_c_result_list = []
    # Wrap the image processing loop with tqdm
    for idx, dct_image_path in enumerate(tqdm(all_dct_image_list, total=len(all_dct_image_list),
                                              desc=f"mode: {mode_str}, Create metrics for DCT", leave=False)):

        # dct_img_index = int(os.path.basename(dct_image_path[0]).split('_')[0])
        dct_counterfeit_img = cv2.imdecode(np.fromfile(dct_image_path[0], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        dct_digital_img = cv2.imdecode(np.fromfile(dct_image_path[1], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        dct_genuine_img = cv2.imdecode(np.fromfile(dct_image_path[2], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        dct_digital_img_ = cv2.resize(dct_digital_img, dsize=dct_genuine_img.shape, interpolation=cv2.INTER_NEAREST)

        dct_genuine_p = utils.pearson(dct_digital_img_, dct_genuine_img)
        dct_genuine_h = utils.hamming_distance(dct_digital_img_, dct_genuine_img)

        dct_counterfeit_p = utils.pearson(dct_digital_img_, dct_counterfeit_img)
        dct_counterfeit_h = utils.hamming_distance(dct_digital_img_, dct_counterfeit_img)

        dct_g_is_decode, dct_g_aeb, dct_g_dc, dct_g_corr_list = utils.dct_decode_result(dct_digital_img,
                                                                                        dct_genuine_img,
                                                                                        args.dct_N, args.dct_alpha,
                                                                                        args.dct_gamma, v)
        dct_c_is_decode, dct_c_aeb, dct_c_dc, dct_c_corr_list = utils.dct_decode_result(dct_digital_img,
                                                                                        dct_counterfeit_img,
                                                                                        args.dct_N, args.dct_alpha,
                                                                                        args.dct_gamma, v)
        if dct_g_is_decode != -2:
            dct_g_result_dict = {"p": dct_genuine_p, "h": dct_genuine_h, "d": dct_g_is_decode, "aeb": dct_g_aeb,
                                 "c": dct_g_corr_list, "dc": dct_g_dc}
            dct_g_result_list.append(dct_g_result_dict)
        if dct_c_is_decode != -2:
            dct_c_result_dict = {"p": dct_counterfeit_p, "h": dct_counterfeit_h, "d": dct_c_is_decode,
                                 "aeb": dct_c_aeb, "c": dct_c_corr_list, "dc": dct_c_dc}
            dct_c_result_list.append(dct_c_result_dict)

    if len(dct_g_result_list) == 0:
        dct_g_result_list = [
            {"p": 0.0, "h": 0.0, "d": -1, "aeb": [0.0, 0.0, 0.0, 0.0], "c": [0.0, 0.0, 0.0],
             "dc": [0.0, 0.0, 0.0]}]
        dct_c_result_list = [
            {"p": 0.0, "h": 0.0, "d": -1, "aeb": [0.0, 0.0, 0.0, 0.0], "c": [0.0, 0.0, 0.0],
             "dc": [0.0, 0.0, 0.0]}]

    metric_dict[f"dct_g"] = dct_g_result_list
    metric_dict[f"dct_c"] = dct_c_result_list

    for idx, cdp_img_path in enumerate(tqdm(all_cdp_image_list, total=len(all_cdp_image_list),
                                            desc=f"mode: {mode_str}, Create metrics for CDP",
                                            leave=False)):
        cdp_counterfeit_img = cv2.imdecode(np.fromfile(cdp_img_path[0], dtype=np.uint8),
                                           cv2.IMREAD_GRAYSCALE)
        cdp_digital_img = cv2.imdecode(np.fromfile(cdp_img_path[1], dtype=np.uint8),
                                       cv2.IMREAD_GRAYSCALE)
        cdp_genuine_img = cv2.imdecode(np.fromfile(cdp_img_path[2], dtype=np.uint8),
                                       cv2.IMREAD_GRAYSCALE)
        cdp_digital_img_ = cv2.resize(cdp_digital_img, dsize=cdp_genuine_img.shape,
                                      interpolation=cv2.INTER_NEAREST)

        cdp_genuine_p = utils.pearson(cdp_digital_img_, cdp_genuine_img)
        cdp_genuine_h = utils.hamming_distance(cdp_digital_img_, cdp_genuine_img)

        cdp_counterfeit_p = utils.pearson(cdp_digital_img_, cdp_counterfeit_img)
        cdp_counterfeit_h = utils.hamming_distance(cdp_digital_img_, cdp_counterfeit_img)
        cdp_g_result_dict = {"p": cdp_genuine_p, "h": cdp_genuine_h}
        cdp_g_result_list.append(cdp_g_result_dict)
        cdp_c_result_dict = {"p": cdp_counterfeit_p, "h": cdp_counterfeit_h}
        cdp_c_result_list.append(cdp_c_result_dict)

    metric_dict[f"cdp_g"] = cdp_g_result_list
    metric_dict[f"cdp_c"] = cdp_c_result_list
    return metric_dict


def compute_pearson_two_level_qrcode_metric(args, mode_str):
    """
    Computes metrics for MSW.

    :param args: Arguments containing lists of N, alpha, and gamma values.
    :param mode_str: Mode of the computation ('pearson' or 'pearson').
    :return: Dictionary containing computed metrics.
    """
    metric_dict = {}
    v = args.scan_ppi // args.print_dpi
    base_path = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
    pearson_path = os.path.join(base_path, args.load_pearson_path.lstrip("/\\"))
    load_pearson_path = os.path.join(pearson_path, mode_str)
    two_level_qrcode_path = os.path.join(base_path, args.load_two_level_qrcode_path.lstrip("/\\"))
    load_two_level_qrcode_path_path = os.path.join(two_level_qrcode_path, mode_str)

    all_pearson_image_list = find_all_files(load_pearson_path)
    all_pearson_image_list.sort()
    all_pearson_image_list = list(group_elements(all_pearson_image_list, 3))
    all_two_level_qrcode_image_list = find_all_files(load_two_level_qrcode_path_path)
    all_two_level_qrcode_image_list.sort()
    all_two_level_qrcode_image_list = list(group_elements(all_two_level_qrcode_image_list, 3))

    pearson_g_result_list = []
    pearson_c_result_list = []
    two_level_qrcode_g_result_list = []
    two_level_qrcode_c_result_list = []
    for idx, pearson_image_path in enumerate(
            tqdm(all_pearson_image_list, total=len(all_pearson_image_list), desc=f"mode: {mode_str}, Create metrics for Pearson", leave=False)):

        pearson_img_index = int(os.path.basename(pearson_image_path[0]).split('_')[0])
        pearson_counterfeit_img = cv2.imdecode(np.fromfile(pearson_image_path[0], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        pearson_digital_img = cv2.imdecode(np.fromfile(pearson_image_path[1], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        pearson_genuine_img = cv2.imdecode(np.fromfile(pearson_image_path[2], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        pearson_digital_img_ = cv2.resize(pearson_digital_img, dsize=pearson_genuine_img.shape,
                                          interpolation=cv2.INTER_NEAREST)

        pearson_genuine_p = utils.pearson(pearson_digital_img_, pearson_genuine_img)
        pearson_genuine_h = utils.hamming_distance(pearson_digital_img_, pearson_genuine_img)

        pearson_counterfeit_p = utils.pearson(pearson_digital_img_, pearson_counterfeit_img)
        pearson_counterfeit_h = utils.hamming_distance(pearson_digital_img_, pearson_counterfeit_img)

        p_g_is_decode, p_g_aeb, p_g_corr_list = utils.pearson_decode_result(pearson_digital_img,
                                                                            pearson_genuine_img,
                                                                            pearson_img_index,
                                                                            args.pearson_N, args.pearson_alpha, args.pearson_gamma, v)
        p_c_is_decode, p_c_aeb, p_c_corr_list = utils.pearson_decode_result(pearson_digital_img,
                                                                            pearson_counterfeit_img,
                                                                            pearson_img_index,
                                                                            args.pearson_N, args.pearson_alpha, args.pearson_gamma, v)
        if p_g_is_decode != -2:
            pearson_g_result_dict = {"p": pearson_genuine_p, "h": pearson_genuine_h, "d": p_g_is_decode, "aeb": p_g_aeb,
                                     "c": p_g_corr_list}
            pearson_g_result_list.append(pearson_g_result_dict)
        if p_g_is_decode != -2:
            pearson_c_result_dict = {"p": pearson_counterfeit_p, "h": pearson_counterfeit_h, "d": p_c_is_decode,
                                     "aeb": p_c_aeb, "c": p_c_corr_list}
            pearson_c_result_list.append(pearson_c_result_dict)

    if len(pearson_g_result_list) == 0:
        pearson_g_result_list = [
            {"p": 0.0, "h": 0.0, "d": -1, "aeb": [0.0, 0.0, 0.0, 0.0], "c": [0.0, 0.0, 0.0]}]
        pearson_c_result_list = [
            {"p": 0.0, "h": 0.0, "d": -1, "aeb": [0.0, 0.0, 0.0, 0.0], "c": [0.0, 0.0, 0.0]}]

    metric_dict[f"pearson_g"] = pearson_g_result_list
    metric_dict[f"pearson_c"] = pearson_c_result_list

    for idx, two_level_qrcode_img_path in enumerate(
            tqdm(all_two_level_qrcode_image_list, total=len(all_two_level_qrcode_image_list),
                 desc=f"mode: {mode_str}, Create metrics for Two Level QRCode",
                 leave=False)):
        two_level_qrcode_counterfeit_img = cv2.imdecode(np.fromfile(two_level_qrcode_img_path[0], dtype=np.uint8),
                                                        cv2.IMREAD_GRAYSCALE)
        two_level_qrcode_digital_img = cv2.imdecode(np.fromfile(two_level_qrcode_img_path[1], dtype=np.uint8),
                                                    cv2.IMREAD_GRAYSCALE)
        two_level_qrcode_genuine_img = cv2.imdecode(np.fromfile(two_level_qrcode_img_path[2], dtype=np.uint8),
                                                    cv2.IMREAD_GRAYSCALE)
        two_level_qrcode_digital_img_ = cv2.resize(two_level_qrcode_digital_img,
                                                   dsize=two_level_qrcode_genuine_img.shape,
                                                   interpolation=cv2.INTER_NEAREST)

        two_level_qrcode_genuine_p = utils.pearson(two_level_qrcode_digital_img_, two_level_qrcode_genuine_img)
        two_level_qrcode_genuine_h = utils.hamming_distance(two_level_qrcode_digital_img_, two_level_qrcode_genuine_img)

        two_level_qrcode_counterfeit_p = utils.pearson(two_level_qrcode_digital_img_, two_level_qrcode_counterfeit_img)
        two_level_qrcode_counterfeit_h = utils.hamming_distance(two_level_qrcode_digital_img_,
                                                                two_level_qrcode_counterfeit_img)

        two_g_aeb, two_g_corr_list = utils.twolqrcode_decode_result(two_level_qrcode_digital_img,
                                                                    two_level_qrcode_genuine_img,
                                                                    args.pearson_N, v)
        two_c_aeb, two_c_corr_list = utils.twolqrcode_decode_result(two_level_qrcode_digital_img,
                                                                    two_level_qrcode_counterfeit_img,
                                                                    args.pearson_N, v)
        two_g_result_dict = {"p": two_level_qrcode_genuine_p, "h": two_level_qrcode_genuine_h, "aeb": two_g_aeb,
                                 "c": two_g_corr_list}
        two_level_qrcode_g_result_list.append(two_g_result_dict)
        two_c_result_dict = {"p": two_level_qrcode_counterfeit_p, "h": two_level_qrcode_counterfeit_h,
                                 "aeb": two_c_aeb, "c": two_c_corr_list}
        two_level_qrcode_c_result_list.append(two_c_result_dict)

    metric_dict[f"two_level_qrcode_g"] = two_level_qrcode_g_result_list
    metric_dict[f"two_level_qrcode_c"] = two_level_qrcode_c_result_list
    return metric_dict


def save_results(args, results, filename):
    """
    Save the results to a specified filename in the 'result' directory.
    :param results: The results to save.
    :param filename: The name of the file to save the results to.
    """
    filepath = os.path.join(args.save_path, filename)
    with open(filepath, 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--print_dpi', type=int, default=600)
    parser.add_argument('--scan_ppi', type=int, default=1200)

    parser.add_argument('--dct_N', type=int, default=48)
    parser.add_argument('--pearson_N', type=int, default=12)
    parser.add_argument('--dct_alpha', type=float, default=0.5)
    parser.add_argument('--pearson_alpha', type=float, default=0.583)
    parser.add_argument('--dct_gamma', type=float, default=0.02)
    parser.add_argument('--pearson_gamma', type=float, default=0.0)

    # Path for original digital binary image
    parser.add_argument('--save_path', type=str, default="metric_results")
    parser.add_argument('--load_dct_path', type=str, default="watermark_effectiveness/datasets_for_comparison/Images/dct/datasets")
    parser.add_argument('--load_pearson_path', type=str, default="watermark_effectiveness/datasets_for_comparison/Images/pearson/datasets")
    parser.add_argument('--load_cdp_path', type=str, default="watermark_effectiveness/datasets_for_comparison/Images/cdp/datasets")
    parser.add_argument('--load_two_level_qrcode_path', type=str, default="watermark_effectiveness/datasets_for_comparison/Images/2lqrcode/datasets")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    scanned_dct_res = compute_cdp_dct_metric(args, "attack_binary")
    save_results(args, scanned_dct_res, 'cdp_dct_attack_binary.json')

    scanned_dct_res = compute_cdp_dct_metric(args, "attack_network")
    save_results(args, scanned_dct_res, 'cdp_dct_attack_network.json')

    scanned_dct_res = compute_pearson_two_level_qrcode_metric(args, "attack_binary")
    save_results(args, scanned_dct_res, 'pearson_two_attack_binary.json')
    #
    scanned_dct_res = compute_pearson_two_level_qrcode_metric(args, "attack_network")
    save_results(args, scanned_dct_res, 'pearson_two_attack_network.json')