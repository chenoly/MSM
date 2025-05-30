import json
import random
import argparse
import time
import numpy as np
from tqdm import tqdm
from models.msw import MSW


def main(args):
    """
    主函数，用于计算和存储MSW框架的错误结果。

    :param args: 参数对象，包括N_list、alpha_list、pearson_gamma_list、dct_gamma_list、Num、seed和save_path等。
    :return: None
    """
    np.random.seed(args.seed)
    random.seed(args.seed)
    result = dict()

    # 计算总的迭代次数
    total_iterations = (len(args.N_list) * len(args.alpha_list) *
                        len(args.pearson_gamma_list) *
                        args.Num)

    with tqdm(total=total_iterations, desc="Processing") as pbar:
        for N in args.N_list:
            model_MSW = MSW(box_size=N)

            for alpha in args.alpha_list:
                for pearson_gamma, dct_gamma in zip(args.pearson_gamma_list, args.dct_gamma_list):
                    dct_list = []
                    pearson_list = []

                    for index in range(args.Num):
                        bits = np.random.binomial(1, 0.5, size=81)

                        start_time = time.time()
                        _ = model_MSW.compute_msw(bits.tolist(), alpha, dct_gamma, "dct", index)
                        end_time = time.time()
                        dct_execution_time = (end_time - start_time)
                        dct_list.append(dct_execution_time)

                        start_time = time.time()
                        _ = model_MSW.compute_msw(bits.tolist(), alpha, pearson_gamma, "pearson", index)
                        end_time = time.time()
                        pearson_execution_time = (end_time - start_time)
                        pearson_list.append(pearson_execution_time)

                        # 更新进度条的描述信息
                        pbar.set_description(
                            f"Processing N={N}, alpha={alpha}, dct_gamma={dct_gamma}, pearson_gamma={pearson_gamma}, index={index}")
                        pbar.update(1)  # 更新进度条

                    result[f"dct_{N}_{alpha}_{dct_gamma}"] = dct_list
                    result[f"dct_{N}_{alpha}_{pearson_gamma}"] = pearson_list

    with open(f"{args.save_path}/time_cost_result.json", 'w') as f:
        json.dump(result, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--Num', type=int, default=10)
    parser.add_argument('--N_list', type=list, default=[12, 24, 36, 48])
    parser.add_argument('--alpha_list', type=list, default=[0.5, 0.6, 0.7])
    parser.add_argument('--pearson_gamma_list', type=list, default=[-0.2, -0.1, 0.0, 0.1, 0.2])
    parser.add_argument('--dct_gamma_list', type=list, default=[0.02, 0.04, 0.06, 0.08, 0.1])
    parser.add_argument('--save_path', type=str, default="result")
    args = parser.parse_args()
    main(args)
