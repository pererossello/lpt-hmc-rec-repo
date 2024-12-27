import sys
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

from forward_model.ics import get_delta
from forward_model.stats import (
    get_pdf,
    get_pow_spec_1D,
    get_cross_power_spec_1D,
    get_reduced_bispectrum,
)
from forward_model.plot_utils import get_projection


def plot_reconstruction(
    savepath,
    u_ref,
    u,
    n_tr_ref,
    n_tr_mean,
    ic_method,
    pow_spec,
    N,
    L,
    N_TR,
    N_SAMPLES,
    ITER,
    ENERGY,
    n_bins,
    extra_text=''
):

    figsize = 6
    ratio = 2.5
    ts = 1.5
    color = "r"
    alpha = 0.5

    width_n_tr = 1
    cmap_n_tr = "gnuplot2"
    lim_n_tr = jnp.mean(n_tr_ref) + jnp.std(n_tr_ref)
    vlim_n_tr = (0, lim_n_tr)

    width_din = 1
    cmap_din = "seismic_r"
    vlim_din = (-1e-1, 1e-1)

    idx, axis = N // 2, 2

    din_ref = get_delta(u_ref, N, L, ic_method, pow_spec, return_hat=False)
    din = get_delta(u, N, L, ic_method, pow_spec, return_hat=False)

    delta_n_tr_ref = N**3 / N_TR * n_tr_ref - 1

    k_cent, pk_in_ref = get_pow_spec_1D(delta=din_ref, n_bins=n_bins, L=L)
    _, pk_n_ref = get_pow_spec_1D(delta=delta_n_tr_ref, n_bins=n_bins, L=L)
    _, propagator = get_cross_power_spec_1D(
        delta_1=din_ref, delta_2=delta_n_tr_ref, n_bins=n_bins, L=L
    )

    fs, rat = figsize, ratio
    fig, axs = plt.subplots(2, 5, figsize=(fs * rat, fs))
    plt.subplots_adjust(wspace=0.25, hspace=0.1)

    n_tr_ref_slice, _, _ = get_projection(
        n_tr_ref, idx=idx, axis=axis, width=width_n_tr
    )
    din_ref_slice, _, _ = get_projection(din_ref, idx=idx, axis=axis, width=width_din)

    axs[0, 0].imshow(
        n_tr_ref_slice.T,
        vmin=vlim_n_tr[0],
        vmax=vlim_n_tr[1],
        origin="lower",
        cmap=cmap_n_tr,
    )
    add_text(axs[0, 0], "N_TR (REF)")

    axs[1, 0].imshow(
        din_ref_slice.T,
        vmin=vlim_din[0],
        vmax=vlim_din[1],
        origin="lower",
        cmap=cmap_din,
    )
    add_text(axs[1, 0], r"$\delta_0$ (REF)")

    for i in range(2):
        for j in range(2):
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])

    zo = 100

    axs[0, 2].plot(
        k_cent, pk_n_ref, c="k", alpha=1, ls="-.", label="N_TR ref", zorder=zo
    )
    axs[0, 2].legend(fontsize=ts * fs, loc="lower left")

    axs[1, 2].plot(
        k_cent, pk_in_ref, c="k", alpha=1, ls="-.", label="DELTA_IN ref", zorder=zo
    )
    axs[1, 2].legend(fontsize=ts * fs, loc="lower left")

    axs[1, 4].plot(
        k_cent, propagator, c="b", alpha=1, ls="-.", label="Propagator", zorder=zo
    )
    axs[1, 4].legend(fontsize=ts * fs, loc="lower left")

    # # din

    din_slice, _, _ = get_projection(din, idx=idx, axis=axis, width=width_din)
    axs[1, 1].imshow(
        din_slice.T, vmin=vlim_din[0], vmax=vlim_din[1], origin="lower", cmap=cmap_din
    )
    add_text(axs[1, 1], r"$\delta_0$ (REC)")

    _, pk_din = get_pow_spec_1D(delta=din, n_bins=n_bins, L=L)
    k_c, c_din = get_cross_power_spec_1D(
        delta_1=din_ref, delta_2=din, n_bins=n_bins, L=L
    )

    axs[1, 2].plot(k_cent, pk_din, c=color, alpha=alpha, ls="-")
    axs[1, 3].plot(k_cent, pk_din / pk_in_ref, c=color, alpha=alpha, ls="-")
    axs[1, 4].plot(k_c, c_din, c=color, alpha=alpha, ls="-")

    # different poisson samples
    key_ = jax.random.PRNGKey(1)
    for i in range(N_SAMPLES):
        key_, _ = jax.random.split(key_)

        n_tr = jax.random.poisson(key_, n_tr_mean)
        delta_n_tr = N**3 / N_TR * n_tr - 1

        if i == 0:
            n_tr_slice, _, _ = get_projection(
                n_tr, idx=idx, axis=axis, width=width_n_tr
            )

            axs[0, 1].imshow(
                n_tr_slice.T,
                vmin=vlim_n_tr[0],
                vmax=vlim_n_tr[1],
                origin="lower",
                cmap=cmap_n_tr,
            )
            add_text(axs[0, 1], "N_TR (REC)")

        # n_tr

        _, pk_n = get_pow_spec_1D(delta=delta_n_tr, n_bins=n_bins, L=L)
        k_c, c_n = get_cross_power_spec_1D(
            delta_1=delta_n_tr_ref, delta_2=delta_n_tr, n_bins=n_bins, L=L
        )

        axs[0, 2].plot(k_cent, pk_n, c=color, alpha=alpha, ls="-")
        axs[0, 3].plot(k_cent, pk_n / pk_n_ref, c=color, alpha=alpha, ls="-")
        axs[0, 4].plot(k_c, c_n, c=color, alpha=alpha, ls="-")

    axs[0, 3].set_ylim(0.5, 1.5)
    axs[1, 3].set_ylim(0.5, 1.5)

    axs[0, 4].set_ylim(-0.1, 1.1)
    axs[1, 4].set_ylim(-0.1, 1.1)

    axs[0, 2].set_yscale("log")
    axs[1, 2].set_yscale("log")
    for i in range(2):
        for j in range(2, 5):
            axs[i, j].set_xscale("log")
            axs[i, j].grid(
                which="major",
                linewidth=0.5,
                color="k",
                alpha=0.25,
            )

    for ax in axs[0, 2:]:
        ax.tick_params(labelbottom=False)

    axs[0, 4].text(
        1.05,
        0.5,
        "N_TR",
        transform=axs[0, 4].transAxes,
        ha="left",
        va="center",
        rotation=90,
    )
    axs[1, 4].text(
        1.05,
        0.5,
        "DELTA_IN",
        transform=axs[1, 4].transAxes,
        ha="left",
        va="center",
        rotation=90,
    )

    x_ = 0.1
    y_ = 1.05
    axs[0, 2].text(
        0.5,
        y_,
        "Pow spec",
        transform=axs[0, 2].transAxes,
        ha="center",
        va="bottom",
    )

    axs[0, 3].text(
        0.5,
        y_,
        "Pow spec rat",
        transform=axs[0, 3].transAxes,
        ha="center",
        va="bottom",
    )

    axs[0, 4].text(
        0.5,
        y_,
        "cross",
        transform=axs[0, 4].transAxes,
        ha="center",
        va="bottom",
    )

    axs[0, 0].text(
        x_,
        y_,
        f"ITER: {ITER} H: {ENERGY:0.2e} {extra_text}",
        transform=axs[0, 0].transAxes,
        ha="left",
        va="bottom",
    )

    axs[0, 1].text(
        x_,
        y_,
        f"range: (0, {lim_n_tr:0.2e})",
        transform=axs[0, 1].transAxes,
        ha="left",
        va="bottom",
    )

    fig.savefig(savepath, bbox_inches="tight")

    plt.close()

    return


def add_text(ax, text, fs=5):
    ax.text(
        0.02,
        0.05,
        text,
        c="k",
        fontsize=fs * 1.7,
        transform=ax.transAxes,
        bbox=dict(
            facecolor="white", edgecolor="black", boxstyle="round,pad=0.1", alpha=0.75
        ),
    )
