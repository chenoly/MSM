import argparse
from models.msg import MSG
from models.msw import MSW
from models.utils import Workflow

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # common parser
    parser.add_argument('--data_len', type=int, default=26)

    parser.add_argument('--N', type=int, default=12)
    parser.add_argument('--alpha', type=float, default=0.583)
    parser.add_argument('--gamma', type=float, default=0.0)

    parser.add_argument('--mode', type=str, default='pearson')    # pearson

    parser.add_argument('--border_size', type=int, default=12)
    parser.add_argument('--inner_border_mm', type=int, default=5)
    parser.add_argument('--out_border_mm', type=int, default=8)
    parser.add_argument('--outer_border_mm', type=int, default=10)

    # print
    parser.add_argument('--Num', type=int, default=1)
    parser.add_argument('--print_dpi', type=int, default=600)
    parser.add_argument('--save_dir', type=str, default=f"Images/pearson/")
    args = parser.parse_args()

    WF = Workflow()
    model = MSW(box_size=args.N, theta1=0.01)
    WF.generate(args.save_dir, args.Num, model, data_len=args.data_len, alpha=args.alpha, gamma=args.gamma, mode=args.mode, print_dpi=args.print_dpi, inner_border_mm=args.inner_border_mm, out_border_mm=args.out_border_mm, outer_border_mm=args.outer_border_mm)
    WF.merge_and_delete_pdfs(f"{args.save_dir}/digital_outer/")