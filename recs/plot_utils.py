import os
import sys
from functools import partial
import h5py
import json

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FixedLocator
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from forward_model.main import get_forward_model
from forward_model.ics import get_delta
from forward_model.stats import (
    get_pdf,
    get_pow_spec_1D,
    get_cross_power_spec_1D,
    get_reduced_bispectrum,
)
from forward_model.plot_utils import get_projection
from forward_model.utils import compute_or_load_pow_spec


class HMCPlot:

    def __init__(self, dir_manager, config, aux_config, fm_config):

        self.fs = 6
        self.ratio = 2.5

        self.ts = 1.75

        self.dir_manager = dir_manager
        self.config = config
        self.aux_config = aux_config

        self.wkd = self.dir_manager.working_directory
        self.exp_dir = self.dir_manager.exp_dir
        self.fig_dirs = self.dir_manager.fig_dirs

        self.N = fm_config["N"]
        self.L = fm_config["L"]
        self.Z_I = fm_config["Z_I"]
        self.N_TR = fm_config["N_TR"]
        self.ic_kind = fm_config["IC_KIND"]

        self.idx = self.N // 2
        self.width = 1
        self.axis = 2
        self.color = "r"
        self.alpha = 1

        self.n_bins = self.aux_config.n_bins

        self.cmap_n_tr = "gnuplot2"
        self.cmap_din = "seismic_r"
        self.vlim_din = (-1e-1, 1e-1)

        fig, axs = plt.subplots(2, 5, figsize=(self.fs * self.ratio, self.fs))
        plt.subplots_adjust(wspace=0.25, hspace=0.1)

        self.fig = fig
        self.axs = axs

        for ax in axs[0, 2:]:
            ax.tick_params(labelbottom=False)
        for i in range(2):
            for j in range(2):
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])

        self.forward_model = get_forward_model(fm_config)
        self.forward_model = jax.jit(self.forward_model)

        self.get_pk_1d = partial(get_pow_spec_1D, n_bins=self.n_bins, L=self.L)
        self.get_pk_1d = jax.jit(self.get_pk_1d)

        self.add_text(self.axs[1, 1], r"$\delta_0$ (REC)")
        self.add_text(axs[0, 1], "N_TR (REC)")

    def plot_sample(self, q, idx_obj):

        axs = self.axs
        color = self.color
        alpha = self.alpha

        pow_spec = self.get_pk()
        din = get_delta(q, self.N, self.L, self.ic_kind, pow_spec, return_hat=False)

        n_tr = self.forward_model(q)
        delta_n_tr = self.N**3 / self.N_TR * n_tr - 1

        din_ref_slice = self.get_proj(din)
        im_n = axs[1, 1].imshow(
            din_ref_slice.T,
            vmin=self.vlim_din[0],
            vmax=self.vlim_din[1],
            origin="lower",
            cmap=self.cmap_din,
        )

        n_tr_slice = self.get_proj(n_tr)
        im_in = axs[0, 1].imshow(
            n_tr_slice.T,
            vmin=self.vlim_n_tr[0],
            vmax=self.vlim_n_tr[1],
            origin="lower",
            cmap=self.cmap_n_tr,
        )

        k_cent, pk_n = self.get_pk_1d(delta=delta_n_tr)
        k_c, c_n = self.get_cross_pk_n_tr(delta_1=delta_n_tr)

        lines_n1 = axs[0, 2].plot(k_cent, pk_n, c=color, alpha=alpha, ls="-")
        lines_n2 = axs[0, 3].plot(
            k_cent, pk_n / self.pk_n_ref, c=color, alpha=alpha, ls="-"
        )
        lines_n3 = axs[0, 4].plot(k_c, c_n, c=color, alpha=alpha, ls="-")

        _, pk_din = self.get_pk_1d(delta=din)
        k_c, c_din = self.get_cross_pk_din(delta_1=din)

        lines_din1 = axs[1, 2].plot(k_cent, pk_din, c=color, alpha=alpha, ls="-")
        lines_din2 = axs[1, 3].plot(
            k_cent, pk_din / self.pk_in_ref, c=color, alpha=alpha, ls="-"
        )
        lines_din3 = axs[1, 4].plot(k_c, c_din, c=color, alpha=alpha, ls="-")

        x_ = 0.1
        y_ = 1.05
        t1 = axs[0, 0].text(
            x_,
            y_,
            f"ITER: {idx_obj.chain_sample_idx:0.0f}",
            transform=axs[0, 0].transAxes,
            ha="left",
            va="bottom",
        )

        T = self.config.temperature[idx_obj.epoch_idx] ** (1 / 3)
        t2 = axs[1, 0].text(
            x_,
            y_,
            f"CH: {idx_obj.chain_idx:0.0f} | EP: {idx_obj.epoch_idx:0.0f} | T: {T:0.1f}^3",
            transform=axs[1, 0].transAxes,
            ha="left",
            va="bottom",
        )

        # Save plot
        fig_dir = self.fig_dirs[idx_obj.chain_idx]
        savepath = os.path.join(fig_dir, f"{idx_obj.plot_idx:04d}.png")
        savepath_2 = os.path.join(self.exp_dir, f"figures/temp.png")
        self.fig.savefig(savepath, bbox_inches="tight")
        self.fig.savefig(savepath_2, bbox_inches="tight")

        # Remove only the new plots (lines and images)
        im_n.remove()
        im_in.remove()

        for line in lines_n1 + lines_n2 + lines_n3:
            line.remove()

        for line in lines_din1 + lines_din2 + lines_din3:
            line.remove()

        t1.remove()
        t2.remove()

    def plot_reference(self):

        savepath = os.path.join(self.wkd, "data_ref.hdf5")
        with h5py.File(savepath, "r") as f:
            n_tr_ref = f["n_tr"][:]
            inp = f["input"][:]

        pow_spec = self.get_pk()

        din_ref = get_delta(
            inp, self.N, self.L, self.ic_kind, pow_spec, return_hat=False
        )

        lim_n_tr = jnp.mean(n_tr_ref) + jnp.std(n_tr_ref)
        self.vlim_n_tr = (0, lim_n_tr)
        n_tr_ref_slice = self.get_proj(n_tr_ref)
        self.axs[0, 0].imshow(
            n_tr_ref_slice.T,
            vmin=self.vlim_n_tr[0],
            vmax=self.vlim_n_tr[1],
            origin="lower",
            cmap=self.cmap_n_tr,
        )
        self.add_text(self.axs[0, 0], "N_TR (REF)")

        din_ref_slice = self.get_proj(din_ref)
        self.axs[1, 0].imshow(
            din_ref_slice.T,
            vmin=self.vlim_din[0],
            vmax=self.vlim_din[1],
            origin="lower",
            cmap=self.cmap_din,
        )
        self.add_text(self.axs[1, 0], r"$\delta_0$ (REF)")

        delta_n_tr_ref = self.N**3 / self.N_TR * n_tr_ref - 1
        self.get_cross_pk_n_tr = partial(
            get_cross_power_spec_1D,
            delta_2=delta_n_tr_ref,
            n_bins=self.n_bins,
            L=self.L,
        )
        self.get_cross_pk_n_tr = jax.jit(self.get_cross_pk_n_tr)

        self.get_cross_pk_din = partial(
            get_cross_power_spec_1D, delta_2=din_ref, n_bins=self.n_bins, L=self.L
        )
        self.get_cross_pk_din = jax.jit(self.get_cross_pk_din)

        k_cent, pk_in_ref = self.get_pk_1d(delta=din_ref)
        _, pk_n_ref = self.get_pk_1d(delta=delta_n_tr_ref)
        _, propagator = self.get_cross_pk_n_tr(delta_1=din_ref)

        self.pk_in_ref = pk_in_ref
        self.pk_n_ref = pk_n_ref

        axs = self.axs
        ts = self.ts
        fs = self.fs

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

        axs[0, 1].text(
            x_,
            y_,
            f"range: (0, {lim_n_tr:0.2e})",
            transform=axs[0, 1].transAxes,
            ha="left",
            va="bottom",
        )

    def get_proj(self, arr):
        arr_slice, _, _ = get_projection(
            arr, idx=self.idx, axis=self.axis, width=self.width
        )
        return arr_slice

    def add_text(self, ax, text, fs=5):
        ax.text(
            0.02,
            0.05,
            text,
            c="k",
            fontsize=fs * self.ts,
            transform=ax.transAxes,
            bbox=dict(
                facecolor="white",
                edgecolor="black",
                boxstyle="round,pad=0.1",
                alpha=0.75,
            ),
        )

    def get_pk(self):
        if self.ic_kind in ["U", "FSK_U"]:
            pow_spec = compute_or_load_pow_spec(self.N, self.L, self.Z_I)
        else:
            pow_spec = None
        return pow_spec


def plot_root_interchain(
    exp_dir, n_bins=20, n_last=5, alpha=0.15, figsize=5, ratio=2.35, show=False
):

    def get_paths_of_samples(exp_dir):
        paths_dic = {}
        for subdir in sorted(os.listdir(exp_dir)):
            if subdir.startswith("root_chain"):
                chain_number = subdir.split("_")[-1]
                samples_file = os.path.join(exp_dir, subdir, "samples.hdf5")
                if os.path.isfile(samples_file):
                    paths_dic[chain_number] = samples_file
        return paths_dic

    rec_dir = os.path.dirname(exp_dir)
    paths_dic = get_paths_of_samples(exp_dir)
    savepath = os.path.join(exp_dir, "figures/root_interchain_plot.png")

    N_CHAINS = len(paths_dic)

    fs = figsize
    ts = 1.25
    cmap = plt.get_cmap("nipy_spectral")
    cs = [cmap(i) for i in np.linspace(0.1, 0.9, N_CHAINS, endpoint=True)]

    fig, axs = interchain_skeleton(figsize=figsize, ratio=ratio, ts=ts*1.2)

    # get forward model
    fm_config_path = os.path.join(rec_dir, "fm_config.json")
    with open(fm_config_path, "r") as json_file:
        fm_config = json.load(json_file)
    N_TR = fm_config["N_TR"]

    forward_model = get_forward_model(fm_config)
    forward_model = jax.jit(forward_model)

    # get ref_data
    ref_data_path = os.path.join(rec_dir, "data_ref.hdf5")
    with h5py.File(ref_data_path, "r") as f:
        N = f["Header"].attrs["N"]
        L = f["Header"].attrs["L"]
        Z_I = f["Header"].attrs["Z_I"]
        IC_KIND = f["Header"].attrs["IC_KIND"]

        n_tr_ref = f["n_tr"][:]
        input_ = f["input"][:]

    delta_n_tr_ref = N**3 / N_TR * n_tr_ref - 1
    if IC_KIND in ["U", "FSK_U"]:
        pow_spec = compute_or_load_pow_spec(N, L, Z_I)
    else:
        pow_spec = None

    din_ref = get_delta(input_, N, L, IC_KIND, pow_spec, return_hat=False)

    get_pk_1d = partial(get_pow_spec_1D, n_bins=n_bins, L=L)
    get_pk_1d = jax.jit(get_pk_1d)

    get_cross_pk_n_tr = partial(
        get_cross_power_spec_1D,
        delta_2=delta_n_tr_ref,
        n_bins=n_bins,
        L=L,
    )
    get_cross_pk_n_tr = jax.jit(get_cross_pk_n_tr)

    get_cross_pk_din = partial(
        get_cross_power_spec_1D, delta_2=din_ref, n_bins=n_bins, L=L
    )
    get_cross_pk_din = jax.jit(get_cross_pk_din)

    k_cent, pk_in_ref = get_pk_1d(delta=din_ref)
    _, pk_n_ref = get_pk_1d(delta=delta_n_tr_ref)
    _, propagator = get_cross_pk_n_tr(delta_1=din_ref)

    zo = 100
    axs[0, 0].plot(
        k_cent, pk_n_ref, c="k", alpha=1, ls="-.", label="N_TR ref", zorder=zo
    )
    axs[0, 0].legend(fontsize=ts * fs, loc="lower left")

    axs[1, 0].plot(
        k_cent, pk_in_ref, c="k", alpha=1, ls="-.", label="DELTA_IN ref", zorder=zo
    )
    axs[1, 0].legend(fontsize=ts * fs, loc="lower left")

    axs[1, 2].plot(
        k_cent, propagator, c="k", alpha=1, ls="-.", label="Propagator", zorder=zo
    )
    axs[1, 2].legend(fontsize=ts * fs, loc="lower left")

    labels = [f"{a}" for a in list(paths_dic.keys())]
    patches = [
        mpatches.Patch(color=color, label=label) for label, color in zip(labels, cs)
    ]


    key_ = jax.random.PRNGKey(1)
    for i, ch_idx in enumerate(paths_dic.keys()):
        key_, _ = jax.random.split(key_)

        path = paths_dic[ch_idx]

        with h5py.File(path, "r") as f:
            f_s = f['samples_00']
            group_names = list(f_s.keys())
            group_names.sort()
            M = int(group_names[-1]) # last group idx

            for j in range(M-n_last, M):
                key_, _ = jax.random.split(key_)

                q = f['samples_00'][f'{j:04d}'][:]

                n_tr_mean = forward_model(q)    
                n_tr = jax.random.poisson(key_, n_tr_mean)
                delta_n_tr = N**3 / N_TR * n_tr - 1     

                _, pk_n = get_pk_1d(delta=delta_n_tr)
                k_c, c_n = get_cross_pk_n_tr(
                    delta_1=delta_n_tr
                )

                axs[0, 0].plot(k_cent, pk_n, c=cs[i], alpha=alpha, ls="-")
                axs[0, 1].plot(k_cent, pk_n / pk_n_ref, c=cs[i], alpha=alpha, ls="-")
                axs[0, 2].plot(k_c, c_n, c=cs[i], alpha=alpha, ls="-")

                din = get_delta(q, N, L, IC_KIND, pow_spec, return_hat=False)
                _, pk_din = get_pk_1d(delta=din)
                _, c_din = get_cross_pk_din(delta_1=din)

                axs[1, 0].plot(k_cent, pk_din, c=cs[i], alpha=alpha, ls="-")
                axs[1, 1].plot(k_cent, pk_din / pk_in_ref, c=cs[i], alpha=alpha, ls="-")
                axs[1, 2].plot(k_c, c_din, c=cs[i], alpha=alpha, ls="-")

    fig.legend(
        handles=patches,
        loc="upper right",
        bbox_to_anchor=(0.9, 0.95),
        ncol=5,
        fontsize=fs * ts,
    )

    fig.savefig(savepath, bbox_inches='tight')

    if show:
        plt.show()

    return


def interchain_skeleton(figsize=5, ratio=2.25, ts=1.25):

    fs, rat = figsize, ratio
    fig, axs = plt.subplots(2, 3, figsize=(fs * rat, fs))
    plt.subplots_adjust(wspace=0.15, hspace=0.15)

    axs[0, 1].set_ylim(0.5, 1.5)
    axs[1, 1].set_ylim(0.5, 1.5)

    axs[0, 2].set_ylim(-0.1, 1.1)
    axs[1, 2].set_ylim(-0.1, 1.1)

    axs[0, 0].set_yscale("log")
    axs[1, 0].set_yscale("log")
    for ax in axs.ravel():
        ax.set_xscale("log")
        ax.grid(
            which="major",
            linewidth=0.5,
            color="k",
            alpha=0.25,
        )

    for ax in axs[0, :]:
        ax.tick_params(labelbottom=False)

    axs[0, 2].text(
        1.05,
        0.5,
        "N_TR",
        transform=axs[0, 2].transAxes,
        ha="left",
        va="center",
        rotation=90,
        fontsize=ts * figsize,
    )
    axs[1, 2].text(
        1.05,
        0.5,
        "DELTA_IN",
        transform=axs[1, 2].transAxes,
        ha="left",
        va="center",
        rotation=90,
        fontsize=ts * figsize,
    )

    # y_ = 1.05
    # axs[0, 0].text(
    #     0.5,
    #     y_,
    #     "Pow spec",
    #     transform=axs[0, 0].transAxes,
    #     ha="center",
    #     va="bottom",
    #     fontsize=ts * figsize,
    # )
    # axs[0, 1].text(
    #     0.5,
    #     y_,
    #     "Pow spec ratio",
    #     transform=axs[0, 1].transAxes,
    #     ha="center",
    #     va="bottom",
    #     fontsize=ts * figsize,
    # )
    # axs[0, 2].text(
    #     0.5, y_, "Cross", transform=axs[0, 2].transAxes, ha="center", va="bottom"
    # )

    return fig, axs







# def plot_statistics_tables(exp_dir):

#     stats_datapath = os.path.join(exp_dir, "stats_00.hdf5")
#     stats_figpath = os.path.join(exp_dir, "figures/stats_00.png")

#     with h5py.File(stats_datapath, "r") as f:
#         grids = {
#             "Mean Times": f["grid_times"][:],
#             "Energy Differences": f["grid_energy_diff"][:],
#             "Acceptance Ratios": f["grid_acc_ratio"][:],
#         }

        

#         n_chains, n_epochs = grids["Mean Times"].shape
#         runtime_str = f['Header'].attrs["total_runtime"][:]

#         ratio = 1.3*(n_chains+1)*(3/4) / (n_epochs+1)

#         # Retrieve additional metadata
#         samples = f["samples"][:]
#         temperatures = f["temperatures"][:]
#         lf_steps = f["lf_steps"][:]
#         lf_step_min = f["lf_step_min"][:]
#         lf_step_max = f["lf_step_max"][:]
        


#         # Create plot
#         fs = 50
#         ts = 7
#         ts_ = ts * 1
#         lw = 0.3
#         vs = 0.19
#         y0 = 1.1
#         y = 1.1
#         font = "monospace"
#         fact_x = jnp.sqrt(fs/ratio)
#         fact_y = fact_x * ratio
#         fig, axes = plt.subplots(4, 1, figsize=(fact_x, fact_y))
#         #plt.subplots_adjust(wspace=0.15, hspace=1*fs)

#         temperatures = temperatures ** (1 / 3)
#         metadata = np.array(
#             [samples, temperatures, lf_steps, lf_step_min, lf_step_max]
#         )
#         metadata_labels = [
#             "N_SAMP",
#             "TEMP*",
#             "N_LF",
#             "LF_MIN",
#             "LF_MAX",
#         ]

#         formats = ["{:.0f}", "{:.1f}", "{:.0f}", "{:.2e}", "{:.2e}"]
#         table_data = [
#             [formats[i].format(metadata[i, j]) for j in range(n_epochs)]
#             for i in range(len(metadata))
#         ]
#         col_labels = [f"EP{i}" for i in range(n_epochs)]
#         row_labels = metadata_labels

#         axes[0].axis("off")
#         metadata_table = axes[0].table(
#             cellText=table_data,
#             rowLabels=row_labels,
#             colLabels=col_labels,
#             cellLoc="center",
#             loc="center",
#         )
#         metadata_table.auto_set_font_size(False)
#         metadata_table.set_fontsize(ts_)
#         axes[0].set_title(
#             "Metadata", fontsize=ts, y=y0, fontfamily=font, color="white"
#         )

#         formats = ["{:.02e}", "{:.2e}", "{:.03f}"]
#         for idx, (title, grid) in enumerate(grids.items()):
#             idx += 1
#             grid_with_means = np.zeros((n_chains + 1, n_epochs + 1))
#             grid_with_means[:-1, :-1] = grid
#             grid_with_means[:-1, -1] = np.mean(grid, axis=1)  # Mean per chain
#             grid_with_means[-1, :-1] = np.mean(grid, axis=0)  # Mean per epoch
#             grid_with_means[-1, -1] = np.mean(grid)  # Overall mean

#             table_data = [
#                 [
#                     formats[idx - 1].format(grid_with_means[i, j])
#                     for j in range(n_epochs + 1)
#                 ]
#                 for i in range(n_chains + 1)
#             ]
#             col_labels = [f"EP{i}" for i in range(n_epochs)] + [""]
#             row_labels = [f"CH{i}" for i in range(n_chains)] + [""]

#             axes[idx].axis("off")
#             table = axes[idx].table(
#                 cellText=table_data,
#                 rowLabels=row_labels,
#                 colLabels=col_labels,
#                 cellLoc="center",
#                 loc="center",
#             )
#             table.auto_set_font_size(False)
#             table.set_fontsize(ts_)
#             axes[idx].set_title(
#                 title, fontsize=ts, y=y, fontfamily=font, color="white"
#             )

#             for key, cell in table.get_celld().items():
#                 cell.set_text_props(color="white")
#                 cell.set_facecolor("black")
#                 cell.set_edgecolor("white")
#                 cell.set_linewidth(lw)
#                 cell.set_text_props(fontfamily=font)
#                 cell.set_height(vs)

#         for key, cell in metadata_table.get_celld().items():
#             cell.set_text_props(color="white")
#             cell.set_facecolor("black")
#             cell.set_edgecolor("white")
#             cell.set_linewidth(lw)
#             cell.set_text_props(fontfamily=font)
#             cell.set_height(vs)

#             # add text to fig
#         fig.text(
#             0.1,
#             0.0,
#             "Runtime: {}".format(runtime_str),
#             ha="center",
#             va="bottom",
#             fontsize=ts,
#             fontfamily=font,
#             color="grey",
#         )

#         for ax in axes:
#             ax.set_facecolor('black')

#         plt.style.use("dark_background")
#         plt.tight_layout(rect=[0, 0, 1, 1])
#         plt.savefig(stats_figpath, bbox_inches="tight")
#         plt.close()


