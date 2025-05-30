import argparse
from models.msg import MSG
from models.msw import MSW
from models.utils import Workflow

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # common parser
    parser.add_argument('--data_len', type=int, default=5, help='seed')

    parser.add_argument('--N', type=int, default=48, help='')
    parser.add_argument('--alpha', type=float, default=0.5, help='')
    parser.add_argument('--gamma', type=float, default=0.02, help='embedding strength')

    parser.add_argument('--mode', type=str, default='dct', help='val_epoch')    # pearson

    parser.add_argument('--border_size', type=int, default=12, help='val_epoch')
    parser.add_argument('--inner_border_mm', type=int, default=5, help='val_epoch')
    parser.add_argument('--out_border_mm', type=int, default=8, help='val_epoch')
    parser.add_argument('--outer_border_mm', type=int, default=10, help='val_epoch')

    # print
    parser.add_argument('--Num', type=int, default=10, help='seed')
    parser.add_argument('--print_dpi', type=int, default=600, help='val_epoch')
    parser.add_argument('--save_dir', type=str, default=f"Images/dct/", help='val_epoch')
    args = parser.parse_args()

    WF = Workflow()
    model = MSG(box_size=args.N)
    WF.generate(args.save_dir, args.Num, model, data_len=args.data_len, alpha=args.alpha, gamma=args.gamma, mode=args.mode, print_dpi=args.print_dpi, inner_border_mm=args.inner_border_mm, out_border_mm=args.out_border_mm, outer_border_mm=args.outer_border_mm)
    WF.merge_and_delete_pdfs(f"{args.save_dir}/digital_outer/")