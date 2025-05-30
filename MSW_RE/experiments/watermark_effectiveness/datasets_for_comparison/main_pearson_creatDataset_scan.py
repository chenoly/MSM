import argparse
from models.msw import MSW
from models.utils import Workflow, generate_random_string

if __name__ == "__main__":
    '''
    定位捕获的MSW
    '''
    print("locate the corner points of the captured MSW...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--outer_border_mm', type=int, default=10, help='val_epoch')
    parser.add_argument('--print_dpi', type=int, default=600, help='val_epoch')
    parser.add_argument('--scan_ppi', type=int, default=1200, help='val_epoch')
    parser.add_argument('--save_captured_located', type=str, default=f"Images/pearson/captured_located", help='val_epoch')
    parser.add_argument('--load_captured_all', type=str, default=f"Images/pearson/captured_all", help='val_epoch')
    parser.add_argument('--save_digital_located', type=str, default=f"Images/pearson/digital_located", help='val_epoch')
    args = parser.parse_args()

    WF = Workflow()
    WF.locate_contours(args.save_captured_located, args.load_captured_all, print_dpi=args.print_dpi, scan_ppi=args.scan_ppi)

    '''
    创建数据集根据定位的MSW
    '''
    print("create the dataset by the located MSW...")

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_len', type=int, default=26, help='seed')
    parser.add_argument('--N', type=int, default=12, help='val_epoch')
    parser.add_argument('--inner_border_mm', type=int, default=5, help='val_epoch')
    parser.add_argument('--out_border_mm', type=int, default=8, help='val_epoch')
    parser.add_argument('--outer_border_mm', type=int, default=10, help='val_epoch')

    # print
    parser.add_argument('--Num', type=int, default=10, help='seed')
    parser.add_argument('--print_dpi', type=int, default=600, help='val_epoch')
    parser.add_argument('--scan_ppi', type=int, default=1200, help='val_epoch')

    # capture
    parser.add_argument('--load_digital_medium', type=str, default=f"Images/pearson/digital_located/", help='')
    parser.add_argument('--load_captured_medium', type=str, default=f"Images/pearson/captured_located/", help='')
    parser.add_argument('--save_dir', type=str, default=f"Images/pearson/genuine/", help='')
    args = parser.parse_args()

    WF = Workflow()
    model = MSW(box_size=args.N)
    marked_code, _ = model.generate_MSG(generate_random_string(args.data_len), 0.5, 0.0)
    WF.createDataset(args.load_digital_medium, args.load_captured_medium, args.save_dir, mgw_size=marked_code.shape, print_dpi=args.print_dpi, scan_ppi=args.scan_ppi)