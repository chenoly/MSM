import os
import argparse
from models.msg import MSG
from models.utils import Workflow, generate_random_string



def locate_points(mode_str):
    """

    :param mode_str:
    :return:
    """

    '''
    定位捕获的MSW
    '''
    parser = argparse.ArgumentParser(description="Argument parser for pearson attack model script.")
    # Border parameters
    parser.add_argument('--outer_border_mm', type=int, default=10, help='Outer border size in millimeters.')
    # Print and scan parameters
    parser.add_argument('--print_dpi', type=int, default=600, help='Print resolution in dots per inch (DPI).')
    parser.add_argument('--scan_ppi', type=int, default=1200, help='Scan resolution in pixels per inch (PPI).')
    # File paths
    parser.add_argument('--load_captured_all', type=str, default="Images/2lqrcode/attack_captured_all")
    parser.add_argument('--load_digital_located', type=str, default="Images/2lqrcode/attacked_medium/attack_located")
    parser.add_argument('--save_captured_located', type=str, default="Images/2lqrcode/attack_captured_located")
    parser.add_argument('--save_captured_medium', type=str, default="Images/2lqrcode/attack_captured_medium")
    parser.add_argument('--save_digital_medium', type=str, default="Images/2lqrcode/attack_digital_medium")
    args = parser.parse_args()

    save_captured_located = os.path.join(args.save_captured_located, mode_str)
    load_captured_all = os.path.join(args.load_captured_all, mode_str)

    WF = Workflow()
    WF.locate_contours(save_captured_located, load_captured_all, print_dpi=args.print_dpi, scan_ppi=args.scan_ppi)



def create_datasets(mode_str):
    """
    创建数据集根据定位的MSW
    :param mode_str:
    :return:
    """
    parser = argparse.ArgumentParser(description="Argument parser for the pearson attack model script.")
    # Common parser arguments
    parser.add_argument('--data_len', type=int, default=86, help='Length of the data.')
    parser.add_argument('--N', type=int, default=12, help='Size of the border in pixels.')
    parser.add_argument('--inner_border_mm', type=int, default=5, help='Inner border size in millimeters.')
    parser.add_argument('--out_border_mm', type=int, default=8, help='Outer border size in millimeters.')
    parser.add_argument('--outer_border_mm', type=int, default=10, help='Outer border size in millimeters.')
    # Print arguments
    parser.add_argument('--print_dpi', type=int, default=600, help='Print resolution in dots per inch (DPI).')
    parser.add_argument('--scan_ppi', type=int, default=1200, help='Scan resolution in pixels per inch (PPI).')
    # Capture arguments
    parser.add_argument('--load_digital_medium', type=str, default="Images/2lqrcode/attacked_medium/attack_located")
    parser.add_argument('--load_captured_medium', type=str, default="Images/2lqrcode/attack_captured_located")
    parser.add_argument('--save_dir', type=str, default="Images/2lqrcode/counterfeit")
    args = parser.parse_args()

    load_digital_medium = os.path.join(args.load_digital_medium, mode_str)
    load_captured_medium = os.path.join(args.load_captured_medium, mode_str)
    save_dir = os.path.join(args.save_dir, mode_str)

    WF = Workflow()
    model = MSG(box_size=args.N)
    marked_code, _ = model.generate_MSG(generate_random_string(args.data_len), 0.5, 0.0)
    WF.createDataset(load_digital_medium, load_captured_medium, save_dir, mgw_size=marked_code.shape, print_dpi=args.print_dpi, scan_ppi=args.scan_ppi)


if __name__ == "__main__":

    locate_points("attack_binary")
    locate_points("attack_identity")
    locate_points("attack_network")

    create_datasets("attack_binary")
    create_datasets("attack_identity")
    create_datasets("attack_network")
