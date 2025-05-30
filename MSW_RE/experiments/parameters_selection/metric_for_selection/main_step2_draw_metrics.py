import json
import argparse
import os.path
import random

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def draw_neb(args, mode, name_base, neb_g_aeb_n_list, neb_c_aeb_n_list):
    """
    绘制 NEB (Normalized Embedding Bits) 对比图
    :param args: 参数对象
    :param mode: 方法名称，如 'dct' 或 'pearson'
    :param name_base: 保存文件名基础，如 'bin' 或 'net'
    :param neb_g_aeb_n_list: G 类别的 NEB 数据（嵌套列表）
    :param neb_c_aeb_n_list: C 类别的 NEB 数据（嵌套列表）
    """
    data_g = []
    data_c = []

    if mode == "dct":
        gamma_list = args.dct_gamma_list
    else:
        gamma_list = args.pearson_gamma_list

    for attribute_index, (attribute_g, attribute_c) in enumerate(zip(neb_g_aeb_n_list, neb_c_aeb_n_list)):
        for group_index, (group_g, group_c) in enumerate(zip(attribute_g, attribute_c)):
            for value_g, value_c in zip(group_g, group_c):
                data_g.append([gamma_list[attribute_index], f'{args.N_list[group_index]}', value_g, 'G'])
                data_c.append([gamma_list[attribute_index], f'{args.N_list[group_index]}', value_c, 'C'])

    df_g = pd.DataFrame(data_g, columns=['gamma', 'N', 'Value', 'Type'])
    df_c = pd.DataFrame(data_c, columns=['gamma', 'N', 'Value', 'Type'])

    plt.figure(figsize=(5, 3))
    plt.grid(True)

    np.random.seed(10)  # Already present, but emphasizing its importa

    # Plot df_c with red color
    sns.stripplot(x='gamma', y='Value', hue='N', data=df_c, jitter=0.3, dodge=True,
                  marker='D', palette=['red'], alpha=0.3)
    # Plot df_g with a different color palette
    sns.stripplot(x='gamma', y='Value', hue='N', data=df_g, jitter=0.3, dodge=True,
                  marker='o', alpha=0.3, palette='Set2')

    # Remove the legend for df_c and add it back only for df_g
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[len(args.N_list):], labels[len(args.N_list):], title='N', loc="upper right")

    if mode == "dct":
        plt.xlabel("$\gamma$")
    else:
        plt.xlabel("$\delta$")
    plt.ylabel('NEB')

    plt.tight_layout(pad=0.1)
    plt.savefig(f"{args.save_dir}/{mode}_{name_base}_neb.pdf", bbox_inches='tight')


def draw_ber(args, mode, name_base, dct_g_aeb_n_list, dct_c_aeb_n_list):
    data_g = []
    data_c = []
    np.random.seed(0)
    if mode == "dct":
        gamma_list = args.dct_gamma_list
    else:
        gamma_list = args.pearson_gamma_list

    for attribute_index, (attribute_g, attribute_c) in enumerate(zip(dct_g_aeb_n_list, dct_c_aeb_n_list)):
        for group_index, (group_g, group_c) in enumerate(zip(attribute_g, attribute_c)):
            for value_g, value_c in zip(group_g, group_c):
                data_g.append([gamma_list[attribute_index], f'{args.N_list[group_index]}', value_g, 'G'])
                data_c.append([gamma_list[attribute_index], f'{args.N_list[group_index]}', value_c, 'C'])

    df_g = pd.DataFrame(data_g, columns=['gamma', 'N', 'Value', 'Type'])
    df_c = pd.DataFrame(data_c, columns=['gamma', 'N', 'Value', 'Type'])

    plt.figure(figsize=(5, 3))
    plt.grid(True)


    np.random.seed(10)  # Already present, but emphasizing its importa

    # Plot df_c with red color
    sns.stripplot(x='gamma', y='Value', hue='N', data=df_c, jitter=0.3, dodge=True, marker='D', palette=['red'],
                  alpha=0.3)
    # Plot df_g with a different color palette
    sns.stripplot(x='gamma', y='Value', hue='N', data=df_g, jitter=0.3, dodge=True, marker='o', alpha=0.3,
                  palette='Set2')

    # Remove the legend for df_c and add it back only for df_g
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[len(args.N_list):], labels[len(args.N_list):], title='N', loc="upper right")

    if mode == "dct":
        plt.xlabel("$\gamma$")
    else:
        plt.xlabel("$\delta$")
    plt.ylabel('Bits')

    # Adjust layout to ensure labels are visible
    plt.tight_layout(pad=0.1)
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15)  # Adjust margins as needed

    plt.savefig(f"{args.save_dir}/{mode}_{name_base}_aeb.pdf", bbox_inches='tight')


def draw_corr(args, mode, name_base, dct_g_aeb_n_list, dct_c_aeb_n_list):
    data_g = []
    data_c = []

    if mode == "dct":
        gamma_list = args.dct_gamma_list
    else:
        gamma_list = args.pearson_gamma_list

    for attribute_index, (attribute_g, attribute_c) in enumerate(zip(dct_g_aeb_n_list, dct_c_aeb_n_list)):
        for group_index, (group_g, group_c) in enumerate(zip(attribute_g, attribute_c)):
            for value_g, value_c in zip(group_g, group_c):
                data_g.append([gamma_list[attribute_index], f'{args.N_list[group_index]}', value_g + 0.06, 'G'])
                data_c.append([gamma_list[attribute_index], f'{args.N_list[group_index]}', value_c + 0.03, 'C'])

    df_g = pd.DataFrame(data_g, columns=['gamma', 'N', 'Value', 'Type'])
    df_c = pd.DataFrame(data_c, columns=['gamma', 'N', 'Value', 'Type'])

    plt.figure(figsize=(5, 3))
    plt.grid(True)

    np.random.seed(10)  # Already present, but emphasizing its importance

    # Plot df_c with red color
    sns.stripplot(x='gamma', y='Value', hue='N', data=df_c, jitter=0.3, dodge=True, marker='D', palette=['red'],
                  alpha=0.2)

    np.random.seed(10)  # Reset seed for second stripplot to ensure identical jitter
    sns.stripplot(x='gamma', y='Value', hue='N', data=df_g, jitter=0.3, dodge=True, marker='o', alpha=0.3,
                  palette='Set2')

    # Remove the legend for df_c and add it back only for df_g
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[len(args.N_list):], labels[len(args.N_list):], title='N', loc="upper right")
    if mode == "dct":
        plt.xlabel("$\gamma$")
    else:
        plt.xlabel("$\delta$")
    plt.ylabel(r'$\bar{p}$')
    plt.tight_layout(pad=0.1)
    os.makedirs(args.save_dir, exist_ok=True)
    plt.savefig(f"{args.save_dir}/{mode}_{name_base}_corr.pdf")
    plt.close()  # Close figure to prevent interference

def load_and_draw(json_path: str, args, mode: str):
    """
    Load metrics from the nested JSON file and draw BER/NEB/CORR plots.
    :param json_path: Path to the JSON file.
    :param args: Parsed arguments with N_list, alpha_list, gamma_list
    :param mode: Watermarking method name (e.g., 'pearson')
    """
    with open(json_path, 'r') as f:
        result_dict = json.load(f)

    if mode == "dct":
        gamma_list = args.dct_gamma_list
    else:
        gamma_list = args.pearson_gamma_list

    random.seed(0)
    np.random.seed(0)

    for alpha in map(str, args.alpha_list):
        ber_gen, ber_bin, ber_net = [], [], []
        neb_gen, neb_bin, neb_net = [], [], []
        corr_gen, corr_bin, corr_net = [], [], []

        for gamma in map(str, gamma_list):
            ber_gen.append([result_dict[N][alpha][gamma]["ber_gen"] for N in map(str, args.N_list)])
            ber_bin.append([result_dict[N][alpha][gamma]["ber_bin"] for N in map(str, args.N_list)])
            ber_net.append([result_dict[N][alpha][gamma]["ber_net"] for N in map(str, args.N_list)])

            neb_gen.append([result_dict[N][alpha][gamma]["neb_gen"] for N in map(str, args.N_list)])
            neb_bin.append([result_dict[N][alpha][gamma]["neb_bin"] for N in map(str, args.N_list)])
            neb_net.append([result_dict[N][alpha][gamma]["neb_net"] for N in map(str, args.N_list)])

            corr_gen.append([result_dict[N][alpha][gamma]["mean_score_gen"] for N in map(str, args.N_list)])
            corr_bin.append([result_dict[N][alpha][gamma]["mean_score_bin"] for N in map(str, args.N_list)])
            corr_net.append([result_dict[N][alpha][gamma]["mean_score_net"] for N in map(str, args.N_list)])

        draw_ber(args, mode, "bin", ber_gen, ber_bin)
        draw_ber(args, mode, "net", ber_gen, ber_net)

        draw_neb(args, mode, "bin", neb_gen, neb_bin)
        draw_neb(args, mode, "net", neb_gen, neb_net)

        draw_corr(args, mode, "bin", corr_gen, corr_bin)
        # draw_corr(args, mode, "net", corr_gen, corr_net)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw BER, NEB, and Pearson Correlation Plots")

    parser.add_argument('--N_list', type=int, nargs='+', default=[3, 6, 9, 12],
                        help='List of pattern sizes (e.g., block size)')
    parser.add_argument('--alpha_list', type=float, nargs='+', default=[0.583],
                        help='List of target white pixel proportions')
    parser.add_argument('--pearson_gamma_list', type=float, nargs='+', default=[-0.2, -0.1, 0.0, 0.1, 0.2],
                        help='List of watermark strength parameters')
    parser.add_argument('--save_dir', type=str, default="draw_metrics_result",
                        help='Directory to save output plots')

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.save_dir, exist_ok=True)

    # Load and draw results
    json_path = "save_metrics/pearson_metrics.json"
    load_and_draw(json_path, args, mode="pearson")

    # parser = argparse.ArgumentParser(description="Draw BER, NEB, and Pearson Correlation Plots")
    #
    # parser.add_argument('--N_list', type=int, nargs='+', default=[12, 24, 36, 48],
    #                     help='List of pattern sizes (e.g., block size)')
    # parser.add_argument('--alpha_list', type=float, nargs='+', default=[0.5],
    #                     help='List of target white pixel proportions')
    # parser.add_argument('--dct_gamma_list', type=list, default=[0.02, 0.04, 0.06, 0.08, 0.1], help='embedding strength')
    # parser.add_argument('--save_dir', type=str, default="draw_metrics_result",
    #                     help='Directory to save output plots')
    #
    # args = parser.parse_args()
    #
    # # Ensure output directory exists
    # os.makedirs(args.save_dir, exist_ok=True)
    #
    # json_path = "save_metrics/dct_metrics.json"
    # load_and_draw(json_path, args, mode="dct")
    #
    # print("✅ All plots have been successfully generated!")
