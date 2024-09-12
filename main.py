import argparse
from applications import DCT_based_MSG, PEARSON_based_MSG

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=12)
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.06)
    parser.add_argument("--q", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument('--data', type=str, default="hello")
    parser.add_argument('--seed', type=int, default=99)
    args = parser.parse_args()
    dct_cdp = DCT_based_MSG(args.N)
    dct_code, bits_embed = dct_cdp.generate_MSG(data=args.data, alpha=args.alpha, gamma=args.gamma, index=args.seed)
    ext_data, ext_bits = dct_cdp.decode(dct_code)
    print("MSW-D CDP", args.data, bits_embed)
    print("MSW-D CDP", ext_data, ext_bits)

    pearson_cdp = PEARSON_based_MSG(args.N)
    pearson_code, bits_embed = pearson_cdp.generate_MSG(data=args.data, alpha=args.alpha, delta=args.delta, q=args.q, index=args.seed)
    ext_data, ext_bits = pearson_cdp.decode(pearson_code, delta=args.delta, q=args.q, index=args.seed)
    print("MSW-P CDP", args.data, bits_embed)
    print("MSW-P CDP", ext_data, ext_bits)
