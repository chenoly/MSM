from models.msg import MSG
from models.parser import args
from models.utils import Workflow, generate_random_string

if __name__ == "__main__":

    WF = Workflow()
    model = MSG(box_size=args.N)
    marked_code, _ = model.generate_MSG(generate_random_string(args.data_len), 0.7, 0.1)
    WF.createDataset(args.digital_match_dir, args.captured_dir, args.save_dir, mgw_size=marked_code.shape, print_dpi=args.print_dpi, scan_ppi=args.scan_ppi)
