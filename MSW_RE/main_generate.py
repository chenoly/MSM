from models.msw import MSW
from models.parser import args
from models.utils import Workflow

if __name__ == "__main__":
    WF = Workflow()
    model = MSW(box_size=args.N, border_size=args.border_size)
    WF.generate(args.save_dir, args.Num, model, data_len=args.data_len, alpha=args.alpha, gamma=args.gamma, mode=args.mode, print_dpi=args.print_dpi, inner_border_mm=args.inner_border_mm, out_border_mm=args.out_border_mm, outer_border_mm=args.outer_border_mm)
