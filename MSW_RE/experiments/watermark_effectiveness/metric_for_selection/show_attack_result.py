import os
import cv2
import random
import argparse
import numpy as np
from matplotlib import pyplot as plt


def random_crop(image, crop_size, x_start, y_start):
    crop = image[y_start:y_start + crop_size, x_start:x_start + crop_size]
    return crop


def process_random_crop(digital_path, attack_path, genuine_path, save_path, name, crop_size=24, step_size=12):
    digital_files = [f for f in os.listdir(digital_path) if f.endswith('.png')]

    # Randomly select a starting position for cropping in the first image
    digital_file = random.choice(digital_files)
    digital_image_path = os.path.join(digital_path, digital_file)
    digital_image = cv2.imdecode(np.fromfile(digital_image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

    attacked_binary_image_path = os.path.join(attack_path, "binattack", f"{digital_file.split('.')[0]}.png")
    attacked_binary_image = cv2.imdecode(np.fromfile(attacked_binary_image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    attacked_binary_image = cv2.resize(attacked_binary_image, dsize=digital_image.shape, interpolation=cv2.INTER_NEAREST)

    attacked_network_image_path = os.path.join(attack_path, "netattack", f"{digital_file.split('.')[0]}.png")
    attacked_network_image = cv2.imdecode(np.fromfile(attacked_network_image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

    # Process genuine image
    genuine_image_path = os.path.join(genuine_path, f"{digital_file.split('.')[0]}.png")
    genuine_image = cv2.imdecode(np.fromfile(genuine_image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

    # Determine valid starting positions based on step_size
    valid_start_positions = range(0, digital_image.shape[1] - crop_size + 1, step_size)
    x_start = random.choice(valid_start_positions)
    y_start = random.choice(valid_start_positions)

    # Perform random cropping on digital image
    digital_crop = random_crop(digital_image, crop_size, x_start, y_start)
    digital_crop_path = os.path.join(save_path, f"{name}_digital_crop.png")
    cv2.imwrite(digital_crop_path, digital_crop)

    attacked_binary_image = random_crop(attacked_binary_image, crop_size, x_start, y_start)
    attacked_binary_crop_path = os.path.join(save_path, f"{name}_binary_crop.png")
    cv2.imwrite(attacked_binary_crop_path, attacked_binary_image)

    attacked_network_crop = random_crop(attacked_network_image, crop_size, x_start, y_start)
    attacked_network_crop_path = os.path.join(save_path, f"{name}_network_crop.png")
    cv2.imwrite(attacked_network_crop_path, attacked_network_crop)

    # Save the cropped genuine image
    genuine_crop = random_crop(genuine_image, crop_size * 2, x_start * 2, y_start * 2)
    genuine_crop_path = os.path.join(save_path, f"{name}_genuine_crop.png")
    cv2.imwrite(genuine_crop_path, genuine_crop)
    # Display and save the images in a row
    image_paths = [digital_crop_path, genuine_crop_path, attacked_binary_crop_path, attacked_network_crop_path]
    return image_paths


def calculate_difference(image1, image2):
    """
    Calculate the difference between two images and return a binary image.
    White pixels (255) indicate differences, black pixels (0) indicate matches.
    """
    diff = np.where(image1 != image2, 255, 0).astype(np.uint8)
    return diff


def display_image_lists(image_lists, save_path=None):
    """
    Displays a list of image paths in a grid with column titles.

    :param image_lists: List of lists, where each inner list contains paths to four images.
    :param save_path: If provided, the function will save the figure as a PDF at this path.
    """
    row_titles = ['MSW-D CDP', 'MSW-P QR code', 'CDP', '2LQR code']
    column_titles = ['Digital', 'Genuine', 'BinAttack', 'BinAttack Diff', 'NetAttack', 'NetAttack Diff']
    plt.rcParams.update({'font.size': 22})
    num_rows = len(image_lists)
    num_cols = len(column_titles)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))

    for i, image_list in enumerate(image_lists):
        images = [cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in image_list]

        # Calculate difference images
        diff_binarized = calculate_difference(images[0], images[2])
        diff_network = calculate_difference(images[0], images[3])

        # Add difference images to the list
        images.insert(3, diff_binarized)
        images.append(diff_network)

        for j, img in enumerate(images):

            ax = axes[i, j] if num_rows > 1 else axes[j]
            ax.imshow(img, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:  # Add titles only to the first row
                ax.set_title(column_titles[j])
            if j == 0:
                ax.set_ylabel(row_titles[i])
            # ax.axis('off')

    plt.tight_layout(pad=0.1, w_pad=2.0, h_pad=0.1)
    if save_path:
        plt.savefig(os.path.join(save_path, "attack_result.pdf"))
        plt.savefig(os.path.join(save_path, "attack_result.png"))
    plt.show()

def main(args):
    random.seed(args.seed)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))

    path_combinations = [
        ('dct', os.path.join(project_root, args.load_digital_dct_path),
         os.path.join(project_root, args.load_attack_dct_path),
         os.path.join(project_root, args.load_genuine_dct_path)),
        ('pearson', os.path.join(project_root, args.load_digital_pearson_path),
         os.path.join(project_root, args.load_attack_pearson_path),
         os.path.join(project_root, args.load_genuine_pearson_path)),
        ('cdp', os.path.join(project_root, args.load_digital_cdp_path),
         os.path.join(project_root, args.load_attack_cdp_path),
         os.path.join(project_root, args.load_genuine_cdp_path)),
        ('two_level_qrcode', os.path.join(project_root, args.load_digital_two_level_qrcode_path),
         os.path.join(project_root, args.load_attack_two_level_qrcode_path),
         os.path.join(project_root, args.load_genuine_two_level_qrcode_path))
    ]

    os.makedirs(args.save_path, exist_ok=True)
    show_img_list = []
    for name, digital_path, attack_path, genuine_path in path_combinations:
        print(f"Processing {name} images for random cropping...")
        print(f"Digital path: {digital_path}")
        print(f"Attack path: {attack_path}")
        print(f"Genuine path: {genuine_path}")
        save_img_list = process_random_crop(digital_path, attack_path, genuine_path, args.save_path, name)
        show_img_list.append(save_img_list)
    display_image_lists(show_img_list, save_path=args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--save_path', type=str, default="save_show_result")
    parser.add_argument('--load_attack_dct_path', type=str,
                        default="datasets_for_comparison/Images/dct/attacked")
    parser.add_argument('--load_digital_dct_path', type=str, default="datasets_for_comparison/Images/dct/digital")
    parser.add_argument('--load_genuine_dct_path', type=str, default="datasets_for_comparison/Images/dct/genuine")

    parser.add_argument('--load_attack_pearson_path', type=str,
                        default="datasets_for_comparison/Images/pearson/attacked")
    parser.add_argument('--load_digital_pearson_path', type=str,
                        default="datasets_for_comparison/Images/pearson/digital")
    parser.add_argument('--load_genuine_pearson_path', type=str,
                        default="datasets_for_comparison/Images/pearson/genuine")
    parser.add_argument('--seed', type=int, default=19)
    parser.add_argument('--load_attack_cdp_path', type=str,
                        default="datasets_for_comparison/Images/cdp/attacked")
    parser.add_argument('--load_digital_cdp_path', type=str, default="datasets_for_comparison/Images/cdp/digital")
    parser.add_argument('--load_genuine_cdp_path', type=str, default="datasets_for_comparison/Images/cdp/genuine")

    parser.add_argument('--load_attack_two_level_qrcode_path', type=str,
                        default="datasets_for_comparison/Images/2lqrcode/attacked")
    parser.add_argument('--load_digital_two_level_qrcode_path', type=str,
                        default="datasets_for_comparison/Images/2lqrcode/digital")
    parser.add_argument('--load_genuine_two_level_qrcode_path', type=str,
                        default="datasets_for_comparison/Images/2lqrcode/genuine")
    args = parser.parse_args()

    main(args)
