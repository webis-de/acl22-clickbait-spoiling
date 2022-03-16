import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

def barplot(plt_tokens, plt_vals, x_label='x', y_label='y', rotation=30, show=True):
    # plt.bar(plt_tokens, plt_vals)
    # plt.xlabel(x_label)
    # plt.ylabel(y_label)
    # plt.xticks(rotation=rotation)
    # plt.grid(axis='y')
    # if show:
    #     plt.show()

    plt.rcdefaults()
    fig, ax = plt.subplots()
    y_pos = np.arange(len(plt_tokens))

    ax.barh(y_pos, plt_vals, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plt_tokens)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if show:
        plt.show()

def boxplot(values_array, labels, x_label='x', y_label='y', width=0.7, show=True):
    ticks = []

    for i, values in enumerate(values_array):
        plt.boxplot(values, widths=width, positions=[i])
        ticks.append(i)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # labels = ['TF-IDF kurz', 'TF-IDF Abschnitt', 'PoS kurz', 'PoS Abschnitt']
    plt.xticks(ticks=ticks, labels=labels)
    plt.grid(axis='y')

    if show:
        plt.show()

def lineplot(plt_vals, x_label='x', y_label='y', rotation=30, show=True):
    plt.plot(plt_vals)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=rotation)
    plt.grid(axis='y')

    if show:
        plt.show()

def multi_lineplot(plt_vals, labels, x_label='x', y_label='y', rotation=30, show=True, smooth_out=1):

    labels_list = list(labels)
    lines = []
    for i, vals in enumerate(plt_vals):
        line = plt.plot(savgol_filter(vals, smooth_out, 3), label=labels_list[i])
        lines.append(line)

    plt.legend(loc='upper right')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=rotation)
    plt.grid(axis='y')

    if show:
        plt.show()