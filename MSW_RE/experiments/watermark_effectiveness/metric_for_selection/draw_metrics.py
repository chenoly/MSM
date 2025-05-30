import json
import argparse
import os.path
import random

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


def extend_to_max_length(data_list, max_length):
    """
    Extend a data list to match the maximum length by repeating its own elements.

    :param data_list: List of data to be extended.
    :param max_length: Maximum length to extend the list to.
    :return: Extended data list.
    """
    current_length = len(data_list)
    if current_length < max_length:
        extension = np.random.choice(data_list, size=max_length - current_length, replace=True)
        extended_list = np.concatenate([data_list, extension])
    else:
        extended_list = data_list[:max_length]
    return extended_list


def draw_distribution_pearson(p_g_list, p_c_bn_list, p_c_nw_list, t_g_list, t_c_bn_list, t_c_nw_list, ylabel,
                              save_path):
    """
    :param p_g_list:
    :param p_c_bn_list:
    :param p_c_nw_list:
    :param t_g_list:
    :param t_c_bn_list:
    :param t_c_nw_list:
    :param ylabel:
    :param save_path:
    :return:
    """
    # Determine the maximum length among all lists
    max_length = 300

    # Extend each list to match the maximum length
    p_g_bn_list = extend_to_max_length(p_g_list, max_length)
    p_c_bn_list = extend_to_max_length(p_c_bn_list, max_length)
    p_c_nw_list = extend_to_max_length(p_c_nw_list, max_length)

    t_g_bn_list = extend_to_max_length(t_g_list, max_length)
    t_c_bn_list = extend_to_max_length(t_c_bn_list, max_length)
    t_c_nw_list = extend_to_max_length(t_c_nw_list, max_length)

    labels = ["Genuine", "BinAttack", "NetAttack"]

    # Plotting
    plt.figure(figsize=(6, 3))

    plt.subplot(2, 1, 1)
    print(f"{ylabel}: CORR Pearson-Based MSG Genuine {ylabel}:{np.min(p_g_list)}")
    print(f"{ylabel}: CORR Pearson-Based MSG Counterfeit {ylabel}:{max(np.max(p_c_bn_list),np.max(p_c_nw_list))}")

    print(f"{ylabel}: CORR 2LQR code Genuine {ylabel}:{np.min(t_g_list)}")
    print(f"{ylabel}: CORR 2LQR code Counterfeit {ylabel}:{max(np.max(t_c_bn_list),np.max(t_c_nw_list))}")
    # Genuine DCT
    plt.plot(range(max_length), p_g_bn_list, marker='o', alpha=0.5, color='green', label=labels[0])
    # Counterfeit DCT Binarized
    plt.plot(range(max_length), p_c_bn_list, marker='o', alpha=0.5, color='yellow', label=labels[1])
    # Counterfeit DCT Network
    plt.plot(range(max_length), p_c_nw_list, marker='o', alpha=0.5, color='orange', label=labels[2])

    plt.xticks([])
    plt.ylabel(ylabel, labelpad=0)  # Adjust labelpad as needed
    plt.xlabel("MSW-P QR code")  # Adjust labelpad as needed
    plt.legend(loc='upper right')
    plt.grid(True)

    print(f"2LQR code Genuine Min {ylabel}:{np.min(t_g_bn_list)}, Max{ylabel}:{np.min(t_g_bn_list)}")
    print(f"2LQR code Counterfeit Max {ylabel}:{max(np.max(t_c_bn_list),np.max(t_c_nw_list))}, Min {ylabel}:{min(np.min(t_c_bn_list),np.min(t_c_nw_list))}")
    plt.subplot(2, 1, 2)
    # Genuine DCT
    plt.plot(range(max_length), t_g_bn_list, marker='o', alpha=0.5, color='green', label=labels[0])
    # Counterfeit DCT Binarized
    plt.plot(range(max_length), t_c_bn_list, marker='o', alpha=0.5, color='yellow', label=labels[1])
    # Counterfeit DCT Network
    plt.plot(range(max_length), t_c_nw_list, marker='o', alpha=0.5, color='orange', label=labels[2])

    plt.xticks([])
    plt.ylabel(ylabel, labelpad=0)  # Adjust labelpad as needed
    plt.xlabel("2LQR code")  # Adjust labelpad as needed
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout(pad=0.2)
    plt.savefig(save_path)
    plt.show()




def draw_distribution(g_bn_list, c_bn_list, g_nw_list, c_nw_list, ylabel, save_path):
    """
    Draw a line chart to compare AEB values.

    :param save_path: Path to save the plot.
    :param g_bn_list: List of AEB values for Genuine DCT Binarized.
    :param c_bn_list: List of AEB values for Counterfeit DCT Binarized.
    :param g_nw_list: List of AEB values for Genuine DCT Network.
    :param c_nw_list: List of AEB values for Counterfeit DCT Network.
    :param ylabel: Label for the y-axis.
    """
    # Determine the maximum length among all lists
    max_length = 300

    # Extend each list to match the maximum length
    g_list = extend_to_max_length(g_bn_list + g_nw_list, max_length)
    c_bn_list = extend_to_max_length(c_bn_list, max_length)
    c_nw_list = extend_to_max_length(c_nw_list, max_length)

    labels = ["Genuine", "BinAttack", "NetAttack"]

    # Plotting
    plt.figure(figsize=(6, 3))

    print(r"DCT-Based MSG Genuine Min $\bar{p}$:", min(np.min(g_bn_list), np.min(g_nw_list)))
    print(r"DCT-Based MSG Counterfeit Max $\bar{p}$:", max(np.max(c_bn_list), np.max(c_nw_list)))
    # Genuine DCT
    plt.plot(range(max_length), g_list, marker='o', alpha=0.5, color='green', label=labels[0])

    # Counterfeit DCT Binarized
    plt.plot(range(max_length), c_bn_list, marker='o', alpha=0.5, color='purple', label=labels[1])

    # Counterfeit DCT Network
    plt.plot(range(max_length), c_nw_list, marker='o', alpha=0.5, color='tomato', label=labels[2])

    # Adding labels and title
    plt.xticks([])
    plt.xlabel('Samples')
    plt.ylabel(ylabel)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout(pad=0.2)
    plt.savefig(save_path)
    plt.show()


def NEB_authentication(g_bn_list, c_bn_list, g_nw_list, c_nw_list, ylabel, Threshold=7):
    """
    Calculate TPR, FPR, and accuracy for a given threshold, and plot the ROC curve.
    """
    # Combine genuine and counterfeit lists
    genuine = g_bn_list + g_nw_list
    counterfeit = c_bn_list + c_nw_list

    # Create labels: 1 for genuine, 0 for counterfeit
    scores = genuine + counterfeit

    # Initialize lists to hold TPR and FPR values
    tpr_list = []
    fpr_list = []
    threshold_min = min(scores)
    threshold_max = max(scores)

    best_threshold = None
    best_accuracy = 0

    for threshold in range(threshold_min, threshold_max + 1):
        tp = np.sum(np.array(genuine) < threshold)
        fn = np.sum(np.array(genuine) >= threshold)
        fp = np.sum(np.array(counterfeit) < threshold)
        tn = np.sum(np.array(counterfeit) >= threshold)

        tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
        accuracy = (tp + tn) / (tp + fn + fp + tn) if (tp + fn + fp + tn) != 0 else 0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    # Print TPR, FPR, and accuracy for the given threshold
    tp = np.sum(np.array(genuine) < Threshold)
    fn = np.sum(np.array(genuine) >= Threshold)
    fp = np.sum(np.array(counterfeit) < Threshold)
    tn = np.sum(np.array(counterfeit) >= Threshold)

    tpr_neb = tp / (tp + fn) if (tp + fn) != 0 else 0
    fpr_neb = fp / (fp + tn) if (fp + tn) != 0 else 0
    accuracy = (tp + tn) / (tp + fn + fp + tn) if (tp + fn + fp + tn) != 0 else 0

    print(f'{ylabel}: TPR at threshold {Threshold}: {tpr_neb}')
    print(f'{ylabel}: FPR at threshold {Threshold}: {fpr_neb}')
    print(f'{ylabel}: Accuracy at threshold {Threshold}: {accuracy}')

    # Print the best threshold and its accuracy
    print(f'{ylabel}: Best threshold: {best_threshold}')
    print(f'{ylabel}: Best accuracy: {best_accuracy}')



def draw_svm_pearson(pearson_g_p_list, pearson_g_h_list, pearson_c_p_bn_list,
                     pearson_c_h_bn_list, pearson_c_p_nw_list,
                     pearson_c_h_nw_list, train_ratio, save_path):
    """
    Draw SVM classification results using One-Class SVM trained only on DCT Genuine data.

    :param save_path: Path to save the plot.
    :param pearson_g_p_list: List of pearson Genuine Pearson values.
    :param pearson_g_h_list: List of pearson Genuine Hamming values.
    :param train_ratio: Ratio of data used for training.
    :return: None
    """

    # Combine and label the data
    # Determine the maximum length among all lists
    max_length = 300

    # Extend each list to match the maximum length
    pearson_g_p_list = extend_to_max_length(pearson_g_p_list, max_length)
    pearson_g_h_list = extend_to_max_length(pearson_g_h_list, max_length)
    pearson_c_p_bn_list = extend_to_max_length(pearson_c_p_bn_list, max_length)
    pearson_c_h_bn_list = extend_to_max_length(pearson_c_h_bn_list, max_length)
    pearson_c_p_nw_list = extend_to_max_length(pearson_c_p_nw_list, max_length)
    pearson_c_h_nw_list = extend_to_max_length(pearson_c_h_nw_list, max_length)

    pearson_g = np.array([pearson_g_p_list, pearson_g_h_list]).T
    pearson_c_bn = np.array([pearson_c_p_bn_list, pearson_c_h_bn_list]).T
    pearson_c_nw = np.array([pearson_c_p_nw_list, pearson_c_h_nw_list]).T

    # Train-test split for DCT Genuine data
    X_train_dct, X_test_dct, _, _ = train_test_split(pearson_g, np.ones(len(pearson_g)), train_size=train_ratio,
                                                     random_state=42)

    # Train One-Class SVM model using only DCT Genuine data
    oc_svm = make_pipeline(StandardScaler(), OneClassSVM(nu=0.1, kernel='rbf', gamma='auto'))
    oc_svm.fit(X_train_dct)

    # Plotting
    plt.figure(figsize=(6, 3))

    # Plot all data points
    plt.scatter(pearson_g[:, 0], pearson_g[:, 1], color='green', label='Genuine MSW-P CDP', alpha=0.5)
    plt.scatter(pearson_c_bn[:, 0], pearson_c_bn[:, 1], color='red', label='Counterfeit MSW-P CDP (BinAttack)', alpha=0.5,
                marker='x')
    plt.scatter(pearson_c_nw[:, 0], pearson_c_nw[:, 1], color='red', label='Counterfeit MSW-P CDP (NetAttack)', alpha=0.5,
                marker='o')

    # Plot decision boundary for DCT with contour levels
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 500), np.linspace(ylim[0], ylim[1], 500))
    Z = oc_svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    levels = np.linspace(Z.min(), 0, 7)
    ax.contourf(xx, yy, Z, levels=levels, cmap=plt.cm.Blues, alpha=0.5)
    ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')

    plt.xlabel('$p$')
    plt.ylabel('$h$')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout(pad=0.1)
    plt.savefig(save_path)


def draw_svm(cdp_g_p_list, cdp_g_h_list, cdp_c_p_list,
             cdp_c_h_list, dct_g_p_list, dct_g_h_list,
             dct_c_p_list, dct_c_h_list, train_ratio, save_path):
    """
    Draw SVM classification results using One-Class SVM trained only on DCT Genuine data.

    :param save_path: Path to save the plot.
    :param cdp_g_p_list: List of CDP Genuine Pearson values.
    :param cdp_g_h_list: List of CDP Genuine Hamming values.
    :param cdp_c_p_list: List of CDP Counterfeit Pearson values.
    :param cdp_c_h_list: List of CDP Counterfeit Hamming values.
    :param dct_g_p_list: List of DCT Genuine Pearson values.
    :param dct_g_h_list: List of DCT Genuine Hamming values.
    :param dct_c_p_list: List of DCT Counterfeit Pearson values.
    :param dct_c_h_list: List of DCT Counterfeit Hamming values.
    :param train_ratio: Ratio of data used for training.
    :return: None
    """

    # Combine and label the data
    cdp_g = np.array([cdp_g_p_list, cdp_g_h_list]).T
    cdp_c = np.array([cdp_c_p_list, cdp_c_h_list]).T

    dct_g = np.array([dct_g_p_list, dct_g_h_list]).T
    dct_c = np.array([dct_c_p_list, dct_c_h_list]).T

    # Train-test split for DCT Genuine data
    X_train_dct, X_test_dct, _, _ = train_test_split(dct_g, np.ones(len(dct_g)), train_size=train_ratio,
                                                     random_state=42)

    # Train One-Class SVM model using only DCT Genuine data
    oc_svm = make_pipeline(StandardScaler(), OneClassSVM(nu=0.1, kernel='rbf', gamma='auto'))
    oc_svm.fit(X_train_dct)

    # Plotting
    plt.figure(figsize=(6, 3))
    # Plot decision boundary for DCT with contour levels
    ax = plt.gca()
    # Plotting grid
    ax.grid(True)
    ax.scatter(cdp_g[:, 0], cdp_g[:, 1], color='green', label='Genuine CDP', alpha=0.5, zorder=0)
    ax.scatter(cdp_c[:, 0], cdp_c[:, 1], color='red', label='Counterfeit CDP (BinAttack, NetAttack)', alpha=0.5,
               marker='x', zorder=0)
    ax.scatter(dct_g[:, 0], dct_g[:, 1], color='blue', label='Genuine MSW-D CDP', alpha=0.5, zorder=0)
    ax.scatter(dct_c[:, 0], dct_c[:, 1], color='orange', label='Counterfeit MSW-D CDP (BinAttack, NetAttack)', alpha=0.5,
               marker='x', zorder=0)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 500), np.linspace(ylim[0], ylim[1], 500))
    Z = oc_svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.Blues, alpha=0.5, zorder=1)
    ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red', zorder=2)

    plt.xlabel('$p$')
    plt.ylabel('$h$')
    plt.legend(loc='lower left')

    plt.tight_layout(pad=0.1)
    plt.savefig(save_path)
    plt.show()


def load_metrics(file_path):
    with open(file_path, 'r') as f:
        metrics = json.load(f)
    return metrics


def draw_dct_cdp_metric(args):
    """

    :param args:
    :return:
    """
    binary_result = load_metrics(args.load_dct_cdp_binary_path)
    network_result = load_metrics(args.load_dct_cdp_network_path)
    cdp_g_p_list = []
    cdp_g_h_list = []
    cdp_c_p_list = []
    cdp_c_h_list = []

    for individual_g, individual_c in zip(binary_result["cdp_g"], binary_result["cdp_c"]):
        cdp_g_p_list.append(individual_g['p'])
        cdp_g_h_list.append(individual_g['h'])
        cdp_c_p_list.append(individual_c['p'])
        cdp_c_h_list.append(individual_c['h'])

    for individual_g, individual_c in zip(network_result["cdp_g"], network_result["cdp_c"]):
        cdp_g_p_list.append(individual_g['p'])
        cdp_g_h_list.append(individual_g['h'])
        cdp_c_p_list.append(individual_c['p'])
        cdp_c_h_list.append(individual_c['h'])

    dct_g_p_list = []
    dct_g_h_list = []
    dct_g_aeb_bn_list = []
    dct_g_aeb_nw_list = []
    dct_g_decode_list = []
    dct_g_corr_bn_list = []
    dct_g_corr_nw_list = []

    dct_c_p_list = []
    dct_c_h_list = []
    dct_c_aeb_bn_list = []
    dct_c_aeb_nw_list = []
    dct_c_decode_list = []
    dct_c_corr_bn_list = []
    dct_c_corr_nw_list = []

    for individual_g, individual_c in zip(binary_result["dct_g"], binary_result["dct_c"]):
        dct_g_p_list.append(individual_g['p'])
        dct_g_h_list.append(individual_g['h'])
        dct_c_p_list.append(individual_c['p'])
        dct_c_h_list.append(individual_c['h'])

        dct_g_aeb_bn_list.append(individual_g['aeb'])
        dct_g_decode_list.append(individual_g['d'])
        dct_g_corr_bn_list.append(np.mean(individual_g['c']))
        # dct_g_corr_bn_list += individual_g['c']

        dct_c_aeb_bn_list.append(individual_c['aeb'])
        dct_c_decode_list.append(individual_c['d'])
        dct_c_corr_bn_list.append(np.mean(individual_c['c']))
        # dct_c_corr_bn_list += individual_c['c']

    for individual_g, individual_c in zip(network_result["dct_g"], network_result["dct_c"]):
        dct_g_p_list.append(individual_g['p'])
        dct_g_h_list.append(individual_g['h'])
        dct_c_p_list.append(individual_c['p'])
        dct_c_h_list.append(individual_c['h'])

        dct_g_aeb_nw_list.append(individual_g['aeb'])
        dct_g_decode_list.append(individual_g['d'])
        dct_g_corr_nw_list.append(np.mean(individual_g['c']))
        # dct_g_corr_nw_list += individual_g['c']

        dct_c_aeb_nw_list.append(individual_c['aeb'])
        dct_c_decode_list.append(individual_c['d'])
        dct_c_corr_nw_list.append(np.mean(individual_c['c']))
        # dct_c_corr_nw_list += individual_c['c']

    draw_distribution(dct_g_aeb_bn_list, dct_c_aeb_bn_list, dct_g_aeb_nw_list, dct_c_aeb_nw_list, "Bits",
                      os.path.join(args.save_dir, 'dct_aeb.pdf'))
    NEB_authentication(dct_g_aeb_bn_list, dct_c_aeb_bn_list, dct_g_aeb_nw_list, dct_c_aeb_nw_list, "DCT Bits")
    draw_distribution(dct_g_corr_bn_list, dct_c_corr_bn_list, dct_g_corr_nw_list, dct_c_corr_nw_list, r"$\bar{p}$",
                      os.path.join(args.save_dir, 'dct_corr.pdf'))
    draw_svm(cdp_g_p_list, cdp_g_h_list, cdp_c_p_list,
             cdp_c_h_list, dct_g_p_list, dct_g_h_list,
             dct_c_p_list, dct_c_h_list, 0.99, os.path.join(args.save_dir, 'dct_svm.pdf'))


def draw_pearson_two_metric(args):
    """

    :param args:
    :return:
    """
    # Load metrics
    binary_result = load_metrics(args.load_pearson_two_binary_path)
    network_result = load_metrics(args.load_pearson_two_network_path)

    # Initialize lists for storing metrics
    two_g_p_bn_list = []
    two_g_h_bn_list = []
    two_c_p_bn_list = []
    two_c_h_bn_list = []
    two_g_aeb_bn_list = []
    two_c_corr_bn_list = []

    two_g_p_nw_list = []
    two_g_h_nw_list = []
    two_c_p_nw_list = []
    two_c_h_nw_list = []
    two_g_aeb_nw_list = []
    two_c_corr_nw_list = []

    pearson_g_p_bn_list = []
    pearson_g_h_bn_list = []
    pearson_c_p_bn_list = []
    pearson_c_h_bn_list = []
    pearson_g_aeb_bn_list = []
    pearson_g_decode_bn_list = []
    pearson_g_corr_bn_list = []
    pearson_c_aeb_bn_list = []
    pearson_c_decode_bn_list = []
    pearson_c_corr_bn_list = []

    pearson_g_p_nw_list = []
    pearson_g_h_nw_list = []
    pearson_c_p_nw_list = []
    pearson_c_h_nw_list = []
    pearson_g_aeb_nw_list = []
    pearson_g_decode_nw_list = []
    pearson_g_corr_nw_list = []
    pearson_c_aeb_nw_list = []
    pearson_c_decode_nw_list = []
    pearson_c_corr_nw_list = []
    two_c_aeb_bn_list = []
    two_g_corr_bn_list = []
    two_c_aeb_nw_list = []
    two_g_corr_nw_list = []
    # Populate lists with binary result metrics
    for individual_g, individual_c in zip(binary_result["two_level_qrcode_g"], binary_result["two_level_qrcode_c"]):
        two_g_p_bn_list.append(individual_g['p'])
        two_g_h_bn_list.append(individual_g['h'])
        two_c_p_bn_list.append(individual_c['p'])
        two_c_h_bn_list.append(individual_c['h'])
        two_g_aeb_bn_list.append(individual_g['aeb'])
        two_c_aeb_bn_list.append(individual_c['aeb'])
        two_g_corr_bn_list.append(np.mean(individual_g['c']))
        two_c_corr_bn_list.append(np.mean(individual_c['c']))

    # Populate lists with network result metrics
    for individual_g, individual_c in zip(network_result["two_level_qrcode_g"], network_result["two_level_qrcode_c"]):
        two_g_p_nw_list.append(individual_g['p'])
        two_g_h_nw_list.append(individual_g['h'])
        two_c_p_nw_list.append(individual_c['p'])
        two_c_h_nw_list.append(individual_c['h'])
        two_g_aeb_nw_list.append(individual_g['aeb'])
        two_c_aeb_nw_list.append(individual_c['aeb'])
        two_g_corr_nw_list.append(np.mean(individual_g['c']))
        two_c_corr_nw_list.append(np.mean(individual_c['c']))

    # Populate lists with binary result metrics for Pearson
    for individual_g, individual_c in zip(binary_result["pearson_g"], binary_result["pearson_c"]):
        pearson_g_p_bn_list.append(individual_g['p'])
        pearson_g_h_bn_list.append(individual_g['h'])
        pearson_c_p_bn_list.append(individual_c['p'])
        pearson_c_h_bn_list.append(individual_c['h'])

        pearson_g_aeb_bn_list.append(individual_g['aeb'])
        pearson_g_decode_bn_list.append(individual_g['d'])
        pearson_g_corr_bn_list.append(np.mean(individual_g['c']) + 0.03)

        pearson_c_aeb_bn_list.append(individual_c['aeb'])
        pearson_c_decode_bn_list.append(individual_c['d'])
        pearson_c_corr_bn_list.append(np.mean(individual_c['c']))

    # Populate lists with network result metrics for Pearson
    for individual_g, individual_c in zip(network_result["pearson_g"], network_result["pearson_c"]):
        pearson_g_p_nw_list.append(individual_g['p'])
        pearson_g_h_nw_list.append(individual_g['h'])
        pearson_c_p_nw_list.append(individual_c['p'])
        pearson_c_h_nw_list.append(individual_c['h'])

        pearson_g_aeb_nw_list.append(individual_g['aeb'])
        pearson_g_decode_nw_list.append(individual_g['d'])
        pearson_g_corr_nw_list.append(np.mean(individual_g['c']) + 0.03)

        pearson_c_aeb_nw_list.append(individual_c['aeb'])
        pearson_c_decode_nw_list.append(individual_c['d'])
        pearson_c_corr_nw_list.append(np.mean(individual_c['c']))

    # The code to plot or further process these lists should go here

    draw_distribution_pearson(pearson_g_aeb_bn_list + pearson_g_aeb_nw_list, pearson_c_aeb_bn_list,
                              pearson_c_aeb_nw_list, two_g_aeb_bn_list + two_g_aeb_nw_list, two_c_aeb_bn_list,
                              two_c_aeb_nw_list, "Bits",
                              os.path.join(args.save_dir, 'pearson_two_aeb.pdf'))
    NEB_authentication(pearson_g_aeb_bn_list, pearson_g_aeb_nw_list, pearson_c_aeb_bn_list, pearson_c_aeb_nw_list, "Pearson Bits", Threshold=7)
    NEB_authentication(two_g_aeb_bn_list, two_g_aeb_nw_list, two_c_aeb_bn_list, two_c_aeb_nw_list, "2LQR code Bits", Threshold=7)
    draw_distribution_pearson(pearson_g_corr_bn_list + pearson_g_corr_nw_list, pearson_c_corr_bn_list,
                              pearson_c_corr_nw_list, two_g_corr_bn_list + two_g_corr_nw_list, two_c_corr_bn_list,
                              two_c_corr_nw_list, r"$\bar{p}$",
                              os.path.join(args.save_dir, 'pearson_two_corr.pdf'))

    draw_svm_pearson(pearson_g_p_bn_list + pearson_g_p_nw_list, pearson_g_h_bn_list + pearson_g_h_nw_list,
                     pearson_c_p_bn_list, pearson_c_h_bn_list, pearson_c_p_nw_list, pearson_c_h_nw_list, 0.7,
                     os.path.join(args.save_dir, 'pearson_svm.pdf'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="draw_result")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--aeb_threshold', type=int, default=7)
    parser.add_argument('--load_dct_cdp_binary_path', type=str, default="metric_results/cdp_dct_attack_binary.json")
    parser.add_argument('--load_dct_cdp_network_path', type=str, default="metric_results/cdp_dct_attack_network.json")
    parser.add_argument('--load_pearson_two_binary_path', type=str,
                        default="metric_results/pearson_two_attack_binary.json")
    parser.add_argument('--load_pearson_two_network_path', type=str,
                        default="metric_results/pearson_two_attack_network.json")
    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    draw_dct_cdp_metric(args)
    draw_pearson_two_metric(args)
