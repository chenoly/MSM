import argparse
import json
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt


def draw(args, alpha, dct_list, pearson_list):
    data_dct = []
    data_pearson = []

    for attribute_index, (attribute_g, attribute_c) in enumerate(zip(dct_list, pearson_list)):
        for group_index, (group_g, group_c) in enumerate(zip(attribute_g, attribute_c)):
            for value_g, value_c in zip(group_g, group_c):
                data_dct.append([args.dct_gamma_list[attribute_index], f'N={args.N_list[group_index]}', value_g, 'G'])
                data_pearson.append(
                    [args.pearson_gamma_list[attribute_index], f'N={args.N_list[group_index]}', value_c, 'C'])

    df_dct = pd.DataFrame(data_dct, columns=['gamma', 'N', 'Value', 'Type'])
    df_pearson = pd.DataFrame(data_pearson, columns=['gamma', 'N', 'Value', 'Type'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5))
    ax1.grid(True)
    ax2.grid(True)

    # Plot for dct_c_aeb_n_list (top plot)
    sns.stripplot(x='gamma', y='Value', hue='N', data=df_dct, jitter=0.3, dodge=True, marker='D', palette='Set2',
                  alpha=0.2, ax=ax1)
    ax1.set_ylabel('Time(s)')
    ax1.set_xticks(range(len(args.dct_gamma_list)))
    ax1.set_xticklabels([f'{x:.2f}' for x in args.dct_gamma_list])
    ax1.set_yticks(ax1.get_yticks())
    ax1.set_yticklabels([f'{y:.2f}' for y in ax1.get_yticks()])
    ax1.set_xlabel(r'$\gamma(\delta)$')

    # Remove the legend for df_c and add it back only for df_g
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[:len(args.N_list)], labels[:len(args.N_list)], title='N', loc="upper left")

    # Plot for dct_g_aeb_n_list (bottom plot)
    sns.stripplot(x='gamma', y='Value', hue='N', data=df_pearson, jitter=0.3, dodge=True, marker='o', alpha=0.3,
                  palette='Set2', ax=ax2)
    ax2.set_ylabel(r'Time(s)')
    ax2.set_xticks(range(len(args.pearson_gamma_list)))
    ax2.set_xticklabels([f'{x:.2f}' for x in args.pearson_gamma_list])
    ax2.set_yticks(ax2.get_yticks())
    ax2.set_yticklabels([f'{y:.2f}' for y in ax2.get_yticks()])
    ax2.set_xlabel('')
    ax2.xaxis.set_ticks_position('top')
    # Remove the legend for df_c and add it back only for df_g
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles[:len(args.N_list)], labels[:len(args.N_list)], title='N', loc="upper center")

    plt.tight_layout(pad=0.1)
    os.makedirs(args.save_path, exist_ok=True)
    plt.savefig(f"{args.save_path}/{alpha}_corr.pdf")


def load_metrics(file_path):
    with open(file_path, 'r') as f:
        metrics = json.load(f)
    return metrics


def main(args):
    """
    主函数，用于计算和存储MSW框架的错误结果。

    :param args: 参数对象，包括N_list、alpha_list、pearson_gamma_list、dct_gamma_list、Num、seed和save_path等。
    :return: None
    """
    np.random.seed(args.seed)
    random.seed(args.seed)
    total_iterations = (len(args.N_list) * len(args.alpha_list) *
                        len(args.pearson_gamma_list))
    result = load_metrics(args.load_path)
    with tqdm(total=total_iterations, desc="Processing") as pbar:
        for alpha in args.alpha_list:
            dct_alpha_list = []
            pearson_alpha_list = []
            for pearson_gamma, dct_gamma in zip(args.pearson_gamma_list, args.dct_gamma_list):
                dct_list = []
                pearson_list = []
                for N in args.N_list:
                    dct_dict = result[f"dct_{N}_{alpha}_{dct_gamma}"]
                    dct_list.append(dct_dict)
                    pearson_dict = result[f"pearson_{N}_{alpha}_{pearson_gamma}"]
                    pearson_list.append(pearson_dict)
                dct_alpha_list.append(dct_list)
                pearson_alpha_list.append(pearson_list)
            draw(args, alpha, dct_alpha_list, pearson_alpha_list)
            # draw_corr(args, alpha, dct_alpha_list, pearson_alpha_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--N_list', type=list, default=[12, 24, 36, 48])
    parser.add_argument('--alpha_list', type=list, default=[0.5, 0.6, 0.7])
    parser.add_argument('--pearson_gamma_list', type=list, default=[-0.2, -0.1, 0.0, 0.1, 0.2])
    parser.add_argument('--dct_gamma_list', type=list, default=[0.02, 0.04, 0.06, 0.08, 0.1])
    parser.add_argument('--save_path', type=str, default="draw_result")
    parser.add_argument('--load_path', type=str, default="result/time_cost_result.json")
    args = parser.parse_args()
    main(args)
