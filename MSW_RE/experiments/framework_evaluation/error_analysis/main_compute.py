import os
import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from models.msg import MSG
from models.msw import MSW


def corr_error(code_1, code_0, gamma):
    """

    :param code_1:
    :param code_0:
    :param N:
    :param gamma:
    :return:
    """
    corr = np.corrcoef(code_1.flatten(), code_0.flatten())[1, 0]
    return abs(abs(corr) - abs(gamma))


def dct_gamma_error(code, bit, model: MSW, gamma):
    """
    
    :param code: 
    :param model: 
    :param N: 
    :param gamma: 
    :return: 
    """

    delta_y = (-1) ** (bit + 1) * gamma * (
            bit * abs(torch.sum(model.template_1 * model.K_x) * (2 / model.box_size)) + (1 - bit) * abs(
        torch.sum(model.template_0 * model.K_x)) * (2 / model.box_size))
    delta_x = np.sum(code * model.K_x.numpy()) * (2 / model.box_size)
    return abs(abs(delta_x) - abs(delta_y))


def alpha_error(code, N, alpha):
    """
    
    :param code: 
    :param alpha: 
    :return: 
    """
    return abs(np.sum(code == 1) / N ** 2 - alpha)


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
            model_MSG = MSG(box_size=N)
            for alpha in args.alpha_list:
                for pearson_gamma, dct_gamma in zip(args.pearson_gamma_list, args.dct_gamma_list):
                    dct_error_alpha_list = []
                    pearson_error_alpha_list = []
                    pearson_error_gamma_list = []
                    dct_error_gamma_list = []

                    for index in range(args.Num):
                        bit = np.random.binomial(1, 0.5, size=1).item()
                        pattern_1_i_t, pattern_0_i_t = model_MSW.compute_pattern(bit=bit, gamma=dct_gamma, alpha=alpha,
                                                                                 mode="dct", seed=index)
                        pattern_i = pattern_1_i_t * model_MSW.template_1.numpy() + pattern_0_i_t * model_MSW.template_0.numpy()
                        one_dct_list = dct_gamma_error(pattern_i, bit, model_MSW, dct_gamma).item()
                        dct_error_gamma_list.append(one_dct_list)
                        one_dct_alpha = alpha_error(pattern_1_i_t, N, alpha).item()
                        dct_error_alpha_list.append(one_dct_alpha)
                        pattern_1_i_t, pattern_0_i_t = model_MSW.compute_pattern(bit=bit, gamma=pearson_gamma,
                                                                                 alpha=alpha, mode="pearson",
                                                                                 seed=index)
                        one_pearson_list = corr_error(pattern_1_i_t, pattern_0_i_t, pearson_gamma).item()
                        pearson_error_gamma_list.append(one_pearson_list)
                        one_pearson_alpha = alpha_error(pattern_1_i_t, N, alpha).item()
                        pearson_error_alpha_list.append(one_pearson_alpha)

                        pbar.set_description(
                            f"Processing N={N}, alpha={alpha}, dct_gamma={dct_gamma}, pearson_gamma={pearson_gamma}, index={index}")
                        pbar.update(1)  # 更新进度条

                    result[f"dct_{N}_{alpha}_{dct_gamma}"] = {"dct_alpha": dct_error_alpha_list,
                                                              "dct_gamma": dct_error_gamma_list}
                    result[f"pearson_{N}_{alpha}_{pearson_gamma}"] = {"pearson_alpha": pearson_error_alpha_list,
                                                                      "pearson_gamma": pearson_error_gamma_list}

    os.makedirs(args.save_path, exist_ok=True)
    with open(f"{args.save_path}/error_result.json", 'w') as f:
        json.dump(result, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--Num', type=int, default=100)
    parser.add_argument('--N_list', type=list, default=[12, 24, 36, 48])
    parser.add_argument('--alpha_list', type=list, default=[0.5, 0.6, 0.7])
    parser.add_argument('--pearson_gamma_list', type=list, default=[-0.2, -0.1, 0.0, 0.1, 0.2])
    parser.add_argument('--dct_gamma_list', type=list, default=[0.02, 0.04, 0.06, 0.08, 0.1])
    parser.add_argument('--save_path', type=str, default="result")
    args = parser.parse_args()
    main(args)
