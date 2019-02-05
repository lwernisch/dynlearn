"""
Plot the results of dynlearn.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_single_sim(x_span, y_span, uvals, y_line, ylim):
    x_out = np.hstack([x_span[:, 1:].T, y_span[-1:, :].T])
    x_out = np.vstack([uvals.T, x_out])
    n_times = x_out.shape[1]

    tls = np.linspace(0, n_times, n_times)
    plt.ylim(ylim)
    plt.plot(tls, x_out[0, :], "c-", label="u")
    plt.plot(tls, x_out[1, :], "b-", label="x1")
    plt.plot(tls, x_out[2, :], "r-", label="x2")
    plt.plot(tls, x_out[3, :], "g-", label="x3")
    plt.axhline(y=y_line, color="m", linewidth=1)
    plt.axvline(x=20, color="k", linewidth=1)
    plt.legend()
    # plt.title('Target: Green at 780')


def plot_sim(i, result_lst, y_line=None, ylim=None, is_subplot=True):
    _, x_span, y_span, uvals = result_lst[i]
    x_out = np.hstack([x_span[:, 1:].T, y_span[-1:, :].T])
    x_out = np.vstack([uvals.T, x_out])
    n_times = x_out.shape[1]

    tls = np.linspace(0, n_times, n_times)
    if is_subplot:
        plt.subplot(2, 3, i + 1)
    plt.ylim(ylim)
    plt.plot(tls, x_out[0, :], "c-")
    plt.plot(tls, x_out[1, :], "b-")
    plt.plot(tls, x_out[2, :], "r-")
    plt.plot(tls, x_out[3, :], "g-")
    if y_line is not None:
        plt.axhline(y=y_line, color="m", linewidth=1)  # , xmin=0.25, xmax=0.75)
    plt.axvline(x=20, color="k", linewidth=1)  # , xmin=0.25, xmax=0.75)
    # plt.title('Target: Green at 780')


def multi_plot(result_lst, y_line=780, ylim=(0, 900)):
    plt.clf()
    for i in range(0, min(6, len(result_lst))):
        plot_sim(i, result_lst, y_line, ylim)
    plt.tight_layout()


def plot_sim_multi_x(i, sim, start, result_lst,
                     y_line, ylim,
                     n_rows, n_cols):
    _, x_span, y_span, uvals = result_lst[i]
    x_out = np.hstack([x_span[:, 1:].T, y_span[-1:, :].T])
    x_out = np.vstack([uvals.T, x_out])
    n_times = x_out.shape[1]
    x_out[0, :] = x_out[0, :] / 4

    tls = np.linspace(1, n_times, n_times)
    plt.subplot(n_rows, n_cols, i + 1 - start)
    variances = ['A'] + sim.output_vars
    plt.ylim(ylim)
    for j in range(x_out.shape[0]):
        plt.plot(tls, x_out[j, :], label=variances[j])
    if i == 0:
        plt.legend()
    plt.axhline(y=y_line, color="m", linewidth=1)
    plt.axvline(x=20, color="k", linewidth=1)
    # plt.title('Target: Green at 780')


def multi_plot_multi_x(sim, result_lst, start=0, y_line=780, ylim=(0, 900),
                       n_rows=2, n_cols=3):
    plt.clf()
    for i in range(start, min(n_cols * n_rows, len(result_lst))):
        plot_sim_multi_x(i, sim, start, result_lst, y_line, ylim,
                         n_rows=n_rows, n_cols=n_cols)
    plt.tight_layout()


def plot_sim_epochs(sim, result_lst):
    n_rows = int(np.floor(np.sqrt(len(result_lst))))
    n_cols = int(np.ceil(len(result_lst) / n_rows))

    plt.figure(figsize=(n_rows * 3, n_cols * 2))
    multi_plot_multi_x(sim, result_lst, start=0, y_line=50, ylim=(0, 85),
                       n_rows=n_rows, n_cols=n_cols)
