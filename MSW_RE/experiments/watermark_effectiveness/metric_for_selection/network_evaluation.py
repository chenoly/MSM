import os
import argparse
import numpy as np
import cv2
import json
import pandas as pd
from tqdm import tqdm


def calculate_error_rate_with_matching(digital_image, attacked_image):
    """
    Calculate the error rate between digital_image and attacked_image using template matching with TM_SQDIFF_NORMED.

    :param digital_image: The reference digital image (template).
    :param attacked_image: The attacked image where we search for matches.
    :return: Error rate as the average of incorrect bits per pixel.
    """
    # Resize attacked_image to match digital_image size
    attacked_image_resized = cv2.resize(attacked_image, (digital_image.shape[1], digital_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Pad attacked_image to ensure matching around borders
    attacked_image_padded = cv2.copyMakeBorder(attacked_image_resized, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)

    # Perform template matching
    result = cv2.matchTemplate(attacked_image_padded, digital_image, cv2.TM_SQDIFF_NORMED)

    # Find minimum squared difference
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Extract matched region coordinates
    match_width, match_height = digital_image.shape[1], digital_image.shape[0]
    match_top_left = min_loc
    match_bottom_right = (match_top_left[0] + match_width, match_top_left[1] + match_height)

    # Extract matched region from padded attacked_image
    matched_region = attacked_image_padded[match_top_left[1]:match_bottom_right[1], match_top_left[0]:match_bottom_right[0]]
    # Calculate error rate as average of incorrect bits per pixel
    error_bits = np.sum(digital_image != matched_region)
    total_bits = digital_image.size
    error_rate = error_bits / total_bits
    return error_rate

def process_images(digital_path, attack_path):
    error_rates_binary = []
    error_rates_network = []
    digital_files = [f for f in os.listdir(digital_path) if f.endswith('.png')]
    for digital_file in tqdm(digital_files, desc="Processing digital images"):
        digital_image_path = os.path.join(digital_path, digital_file)
        digital_image = cv2.imdecode(np.fromfile(digital_image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        attacked_image_binary_path = os.path.join(attack_path, "binattack", f"{digital_file.split('.')[0]}.png")
        attacked_binary_image = cv2.imdecode(np.fromfile(attacked_image_binary_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        attacked_image_network_path = os.path.join(attack_path, "netattack", f"{digital_file.split('.')[0]}.png")
        attacked_network_image = cv2.imdecode(np.fromfile(attacked_image_network_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        error_binary_rate = calculate_error_rate_with_matching(digital_image, attacked_binary_image)
        error_network_rate = calculate_error_rate_with_matching(digital_image, attacked_network_image)
        error_rates_binary.append(error_binary_rate)
        error_rates_network.append(error_network_rate)
    return {"binary": error_rates_binary, "network": error_rates_network}

def generate_error_rate_metrics(args):
    results = {}

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))

    path_combinations = [
        ('dct', os.path.join(project_root, args.load_digital_dct_path),
         os.path.join(project_root, args.load_attack_dct_path)),
        ('pearson', os.path.join(project_root, args.load_digital_pearson_path),
         os.path.join(project_root, args.load_attack_pearson_path)),
        ('cdp', os.path.join(project_root, args.load_digital_cdp_path),
         os.path.join(project_root, args.load_attack_cdp_path)),
        ('two_level_qrcode', os.path.join(project_root, args.load_digital_two_level_qrcode_path),
         os.path.join(project_root, args.load_attack_two_level_qrcode_path))
    ]

    for name, digital_path, attack_path in path_combinations:
        print(f"Processing {name} images...")
        print(f"Digital path: {digital_path}")
        print(f"Attack path: {attack_path}")
        error_rates = process_images(digital_path, attack_path)
        results[name] = error_rates

    os.makedirs(args.save_path, exist_ok=True)
    with open(os.path.join(args.save_path, 'network_error_rates.json'), 'w') as f:
        json.dump(results, f, indent=4)


def save_error_rates_as_latex(args):
    """

    :param args:
    :return:
    """
    json_file = os.path.join(args.save_path, 'network_error_rates.json')
    latex_file = os.path.join(args.save_path, 'table.tex')
    with open(json_file, 'r') as f:
        data = json.load(f)
    average_error_rates = {'Method': [], 'Binaried': [], 'Network': []}
    for method, error_rates in data.items():
        average_error_rates['Method'].append(method)
        average_error_rates['Binaried'].append(np.mean(error_rates['binary']))
        average_error_rates['Network'].append(np.mean(error_rates['network']))
    df = pd.DataFrame(average_error_rates)
    latex_table = df.to_latex(index=False)
    with open(latex_file, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table saved to {latex_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--save_path', type=str, default="metric_results")
    parser.add_argument('--load_attack_dct_path', type=str, default="datasets_for_comparison/Images/dct/attacked")
    parser.add_argument('--load_digital_dct_path', type=str, default="datasets_for_comparison/Images/dct/digital")
    parser.add_argument('--load_attack_pearson_path', type=str, default="datasets_for_comparison/Images/pearson/attacked")
    parser.add_argument('--load_digital_pearson_path', type=str, default="datasets_for_comparison/Images/pearson/digital")
    parser.add_argument('--load_attack_cdp_path', type=str, default="datasets_for_comparison/Images/cdp/attacked")
    parser.add_argument('--load_digital_cdp_path', type=str, default="datasets_for_comparison/Images/cdp/digital")
    parser.add_argument('--load_attack_two_level_qrcode_path', type=str, default="datasets_for_comparison/Images/2lqrcode/attacked")
    parser.add_argument('--load_digital_two_level_qrcode_path', type=str, default="datasets_for_comparison/Images/2lqrcode/digital")
    args = parser.parse_args()
    generate_error_rate_metrics(args)
    save_error_rates_as_latex(args)
