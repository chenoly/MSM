import os
import json
import argparse
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


# def draw(args, alpha, mode, dct_list, pearson_list):
#     data_dct = []
#     data_pearson = []
#
#     for attribute_index, (attribute_g, attribute_c) in enumerate(zip(dct_list, pearson_list)):
#         for group_index, (group_g, group_c) in enumerate(zip(attribute_g, attribute_c)):
#             for value_g, value_c in zip(group_g, group_c):
#                 data_dct.append([args.dct_gamma_list[attribute_index], f'{args.N_list[group_index]}', value_g, 'G'])
#                 data_pearson.append(
#                     [args.pearson_gamma_list[attribute_index], f'{args.N_list[group_index]}', value_c, 'C'])
#
#     df_dct = pd.DataFrame(data_dct, columns=['gamma', 'N', 'Value', 'Type'])
#     df_pearson = pd.DataFrame(data_pearson, columns=['gamma', 'N', 'Value', 'Type'])
#
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5))
#     ax1.grid(True)
#     ax2.grid(True)
#     # Plot for dct_c_aeb_n_list (top plot)
#     sns.stripplot(x='gamma', y='Value', hue='N', data=df_dct, jitter=0.3, dodge=True, marker='D', palette='Set3',
#                   alpha=0.2, ax=ax1)
#     ax1.set_ylabel(r'Error ($\alpha$)')
#     ax1.set_xticks(range(len(args.dct_gamma_list)))
#     ax1.set_xticklabels(args.dct_gamma_list)
#     ax1.set_xlabel(r'$\gamma$')
#
#     # Remove the legend for df_c and add it back only for df_g
#     handles, labels = ax1.get_legend_handles_labels()
#     ax1.legend(handles[:len(args.N_list)], labels[:len(args.N_list)], title='N', loc="upper right")
#
#     # Plot for dct_g_aeb_n_list (bottom plot)
#     sns.stripplot(x='gamma', y='Value', hue='N', data=df_pearson, jitter=0.3, dodge=True, marker='o', alpha=0.3,
#                   palette='Set2', ax=ax2)
#     ax2.set_ylabel(r'Error ($\delta$)')
#     ax2.set_xticks(range(len(args.pearson_gamma_list)))
#     ax2.set_xticklabels(args.pearson_gamma_list)
#     ax2.set_xlabel(r'$\gamma$')
#
#     # Remove the legend for df_c and add it back only for df_g
#     handles, labels = ax2.get_legend_handles_labels()
#     ax2.legend(handles[:len(args.N_list)], labels[:len(args.N_list)], title='N', loc="upper right")
#
#     plt.tight_layout(pad=0.1)
#     os.makedirs(args.save_path, exist_ok=True)
#     plt.savefig(f"{args.save_path}/{mode}_{alpha}_corr.pdf")



def draw(args, alpha, dct_alpha_list, dct_gamma_list, pearson_alpha_list, pearson_gamma_list):
    data_dct_alpha = []
    data_pearson_alpha = []
    data_dct_gamma = []
    data_pearson_gamma = []

    for attribute_index, (attribute_d_alpha, attribute_p_alpha, attribute_d_gamma, attribute_p_gamma) in enumerate(
            zip(dct_alpha_list, pearson_alpha_list, dct_gamma_list, pearson_gamma_list)):
        for group_index, (group_d_alpha, group_p_alpha, group_d_gamma, group_p_gamma) in enumerate(
                zip(attribute_d_alpha, attribute_p_alpha, attribute_d_gamma, attribute_p_gamma)):
            for value_d_alpha, value_p_alpha, value_d_gamma, value_p_gamma in zip(group_d_alpha, group_p_alpha,
                                                                                  group_d_gamma, group_p_gamma):
                data_dct_alpha.append(
                    [args.dct_gamma_list[attribute_index], f'N={args.N_list[group_index]}', value_d_alpha, 'G'])
                data_pearson_alpha.append(
                    [args.pearson_gamma_list[attribute_index], f'N={args.N_list[group_index]}', value_p_alpha, 'C'])
                data_dct_gamma.append(
                    [args.dct_gamma_list[attribute_index], f'N={args.N_list[group_index]}', value_d_gamma, 'G'])
                data_pearson_gamma.append(
                    [args.pearson_gamma_list[attribute_index], f'N={args.N_list[group_index]}', value_p_gamma, 'C'])

    df_dct_alpha = pd.DataFrame(data_dct_alpha, columns=['gamma', 'N', 'Value', 'Type'])
    df_pearson_alpha = pd.DataFrame(data_pearson_alpha, columns=['gamma', 'N', 'Value', 'Type'])
    df_dct_gamma = pd.DataFrame(data_dct_gamma, columns=['gamma', 'N', 'Value', 'Type'])
    df_pearson_gamma = pd.DataFrame(data_pearson_gamma, columns=['gamma', 'N', 'Value', 'Type'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5))

    # Top plot
    ax1.grid(True)
    ax1_2 = ax1.twiny()
    sns.stripplot(x='gamma', y='Value', hue='N', data=df_dct_alpha, jitter=0.3, dodge=True, marker='D', palette='Set3',
                  alpha=0.6, ax=ax1)
    sns.stripplot(x='gamma', y='Value', hue='N', data=df_pearson_alpha, jitter=0.3, dodge=True, marker='o',
                  palette='Set1', alpha=0.6)
    ax1.set_ylabel(r"")
    ax1.set_xlabel(r'$\gamma(\delta)$')

    ax1.set_xticks(range(len(args.dct_gamma_list)))
    ax1.set_xticklabels([f'{x:.2f}' for x in args.dct_gamma_list])
    ax1_2.set_xticks(range(len(args.pearson_gamma_list)))
    ax1_2.set_xticklabels([f'{x:.2f}' for x in args.pearson_gamma_list])


    # Bottom plot
    ax2.grid(True)
    ax2_2 = ax2.twiny()
    sns.stripplot(x='gamma', y='Value', hue='N', data=df_dct_gamma, jitter=0.3, dodge=True, marker='D', palette='Set3',
                  alpha=0.6, ax=ax2)
    sns.stripplot(x='gamma', y='Value', hue='N', data=df_pearson_gamma, jitter=0.3, dodge=True, marker='o',
                  palette='Set1', alpha=0.6)
    ax2.set_ylabel(r'')
    ax2.set_xlabel(r'$\gamma(\delta)$')

    ax2.set_xticks(range(len(args.dct_gamma_list)))
    ax2.set_xticklabels([f'{x:.2f}' for x in args.dct_gamma_list])
    ax2_2.set_xticks(range(len(args.pearson_gamma_list)))
    ax2_2.set_xticklabels([f'{x:.2f}' for x in args.pearson_gamma_list])

    # Add legends
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax1_2.get_legend_handles_labels()
    ax1.legend(handles1, labels1, title='DCT', loc="upper right")
    ax1_2.legend(handles2, labels2, title='Pearson', loc="upper right", bbox_to_anchor=(0.8, 1))
    ax1_2.set_xlabel('')
    # Add legends
    handles1, labels1 = ax2.get_legend_handles_labels()
    handles2, labels2 = ax2_2.get_legend_handles_labels()
    ax2.legend(handles1, labels1, title='DCT', loc="upper right", bbox_to_anchor=(1, 0.75))
    ax2_2.legend(handles2, labels2, title='Pearson', loc="upper right", bbox_to_anchor=(0.8, 0.75))
    ax2_2.set_xlabel('')
    ax2.set_xlabel('')

    plt.tight_layout(pad=0.1)
    os.makedirs(args.save_path, exist_ok=True)
    plt.savefig(f"{args.save_path}/{alpha}_corr.pdf")



def main(args):
    """

    :param args:
    :return:
    """
    with open(args.load_path, 'r') as f:
        result_dict = json.load(f)
    for alpha in args.alpha_list:
        dct_A_alpha_list = []
        dct_A_gamma_list = []

        pearson_A_alpha_list = []
        pearson_A_gamma_list = []
        for pearson_gamma, dct_gamma in zip(args.pearson_gamma_list, args.dct_gamma_list):

            dct_N_alpha_list = []
            dct_N_gamma_list = []

            pearson_N_alpha_list = []
            pearson_N_gamma_list = []
            for N in args.N_list:
                dct_dict = result_dict[f"dct_{N}_{alpha}_{dct_gamma}"]
                dct_N_alpha_list.append(dct_dict['dct_alpha'])
                dct_N_gamma_list.append(dct_dict['dct_gamma'])

                pearson_dict = result_dict[f"pearson_{N}_{alpha}_{pearson_gamma}"]
                pearson_N_alpha_list.append(pearson_dict['pearson_alpha'])
                pearson_N_gamma_list.append(pearson_dict['pearson_gamma'])

            dct_A_alpha_list.append(dct_N_alpha_list)
            dct_A_gamma_list.append(dct_N_gamma_list)

            pearson_A_alpha_list.append(pearson_N_alpha_list)
            pearson_A_gamma_list.append(pearson_N_gamma_list)

        draw(args, alpha, dct_A_alpha_list, dct_A_gamma_list, pearson_A_alpha_list, pearson_A_gamma_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--N_list', type=list, default=[12, 24, 36, 48])
    parser.add_argument('--alpha_list', type=list, default=[0.5, 0.6, 0.7])
    parser.add_argument('--pearson_gamma_list', type=list, default=[-0.2, -0.1, 0.0, 0.1, 0.2])
    parser.add_argument('--dct_gamma_list', type=list, default=[0.02, 0.04, 0.06, 0.08, 0.1])
    parser.add_argument('--save_path', type=str, default="draw_result")
    parser.add_argument('--load_path', type=str, default="result/error_result.json")
    args = parser.parse_args()
    main(args)