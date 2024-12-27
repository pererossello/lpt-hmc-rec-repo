import os
import sys
from functools import partial
import h5py

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

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

        T = self.config.temperature[idx_obj.epoch_idx]**(1/3)
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
