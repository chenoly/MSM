import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from models.attacker import BinAttack
import json


def evaluate_binary(genuine_code, digital_code, radius, amount, binattack):
    """
    Evaluates the binarization performance for a single image pair using SSIM and BER.

    :param genuine_code: PIL Image, scanned grayscale image
    :param digital_code: PIL Image, original digital grayscale image
    :param radius: Radius for unsharp mask
    :param amount: Strength for unsharp mask
    :param binattack: BinAttack instance
    :return: Tuple of (SSIM score, BER score)
    """
    genuine = np.array(genuine_code).astype(float) / 255.0
    digital = np.array(digital_code).astype(float) / 255.0
    binary_img = binattack.binary(genuine * 255., unsharp_radius=radius, unsharp_amount=amount)

    plt.subplot(131)
    plt.imshow(binary_img, cmap='gray')
    plt.subplot(132)
    plt.imshow(genuine, cmap='gray')
    plt.subplot(132)
    plt.imshow(digital, cmap='gray')
    plt.title(f"radius:{radius}, amount{amount}")
    plt.show()

    ssim_score = ssim(binary_img, digital, data_range=1.0)
    ber_score = np.mean(np.abs(binary_img - digital))
    return ssim_score, ber_score


def optimize_parameters_all_images(args, binattack):
    """
    Optimizes unsharp mask parameters (radius, amount) using all images in the dataset.
    Computes average SSIM and BER across all images for each parameter combination.

    :param args: Command-line arguments containing directory paths and parameter lists
    :param binattack: BinAttack instance
    :return: Tuple of (best parameters, list of results)
    """
    radius_list = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    amount_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    ssim_scores = np.zeros((len(radius_list), len(amount_list)))
    ber_scores = np.zeros((len(radius_list), len(amount_list)))
    image_count = 0
    results = []

    # Iterate over all N, alpha, gamma combinations
    for N in args.N_list:
        N_path_genuine = os.path.join(args.genuine_dir, f"N_{N}")
        N_path_digital = os.path.join(args.digital_dir, f"N_{N}")
        for alpha in args.alpha_list:
            alpha_path_genuine = os.path.join(N_path_genuine, f"alpha_{alpha}")
            alpha_path_digital = os.path.join(N_path_digital, f"alpha_{alpha}")
            for gamma in args.gamma_list:
                gamma_path_genuine = os.path.join(alpha_path_genuine, f"gamma_{gamma}")
                gamma_path_digital = os.path.join(alpha_path_digital, f"gamma_{gamma}")
                genuine_image_files = [f for f in os.listdir(gamma_path_genuine)
                                       if f.lower().endswith((".png", ".jpg", ".bmp", ".jpeg"))]
                digital_image_files = [f for f in os.listdir(gamma_path_digital)
                                       if f.lower().endswith((".png", ".jpg", ".bmp", ".jpeg"))]

                # Process each image pair
                for genuine_file, digital_file in zip(genuine_image_files, digital_image_files):
                    genuine_path = os.path.join(gamma_path_genuine, genuine_file)
                    digital_path = os.path.join(gamma_path_digital, digital_file)

                    # Load images
                    try:
                        genuine_code = Image.open(genuine_path).convert('L')
                        digital_code = Image.open(digital_path).convert('L').resize(genuine_code.size)
                    except Exception as e:
                        print(f"Error loading {genuine_path} or {digital_path}: {e}")
                        continue

                    # Evaluate each parameter combination
                    for i, radius in enumerate(radius_list):
                        for j, amount in enumerate(amount_list):
                            ssim_score, ber_score = evaluate_binary(
                                genuine_code, digital_code, radius, amount, binattack
                            )
                            ssim_scores[i, j] += ssim_score
                            ber_scores[i, j] += ber_score
                    image_count += 1
                    print(f"Processed {genuine_file} (N={N}, alpha={alpha}, gamma={gamma})")

    # Compute average scores
    if image_count > 0:
        ssim_scores /= image_count
        ber_scores /= image_count
    else:
        raise ValueError("No images processed. Check directory paths or file formats.")

    # Store results for all parameter combinations
    for i, radius in enumerate(radius_list):
        for j, amount in enumerate(amount_list):
            results.append({
                'radius': radius,
                'amount': amount,
                'ssim': ssim_scores[i, j],
                'ber': ber_scores[i, j]
            })

    # Find best parameters based on maximum SSIM
    best_idx = np.unravel_index(np.argmin(ber_scores), ber_scores.shape)
    best_params = (radius_list[best_idx[0]], amount_list[best_idx[1]])

    print(f"Best Parameters: Radius={best_params[0]}, Amount={best_params[1]}, "
          f"SSIM={ssim_scores[best_idx]:.4f}, BER={ber_scores[best_idx]:.4f}")

    return best_params, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_len', type=int, default=81)
    parser.add_argument('--N_list', type=list, default=[12])
    parser.add_argument('--alpha_list', type=list, default=[0.583])
    parser.add_argument('--gamma_list', type=list, default=[0.0])
    parser.add_argument('--border_size', type=int, default=12)
    parser.add_argument('--inner_border_mm', type=int, default=5)
    parser.add_argument('--out_border_mm', type=int, default=8)
    parser.add_argument('--outer_border_mm', type=int, default=10)
    parser.add_argument('--print_dpi', type=int, default=600)
    parser.add_argument('--scan_ppi', type=int, default=1200)
    parser.add_argument('--digital_dir', type=str, default="Images/pearson/digital/")
    parser.add_argument('--genuine_dir', type=str, default="Images/pearson/genuine/")
    parser.add_argument('--save_netattacked_dir', type=str, default="Images/pearson/counterfeit/NetAttack")
    parser.add_argument('--save_binattacked_dir', type=str, default="Images/pearson/counterfeit/BinAttack")
    args = parser.parse_args()

    binattack = BinAttack(args.print_dpi, args.scan_ppi)

    # Optimize parameters using all images
    best_params, results = optimize_parameters_all_images(args, binattack)

    # Save results to JSON
    with open("unsharp_param_results_all.json", "w") as f:
        json.dump(results, f, indent=4)

    # Optionally save binarized images for visual inspection
    # save_binary_images_all(args, binattack)