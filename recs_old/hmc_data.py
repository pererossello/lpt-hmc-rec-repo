import h5py
import os
import shutil
import json

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import jax.numpy as jnp


class HMCData:

    def __init__(self, path):

        self.path = path
        self._load_header_attributes()

    def _load_header_attributes(self):
        with h5py.File(self.path, "r") as f:
            # Access the "Header" group
            header = f["Header"]
            for key, value in header.attrs.items():
                setattr(self, key.lower(), value)  # Lowercase for consistency

    def copy_to(self, target_directory, basename):
        # Ensure target directory exists
        os.makedirs(target_directory, exist_ok=True)

        # Construct new file path
        filename = f"{basename}.hdf5"
        target_path = os.path.join(target_directory, filename)

        # Perform the copy operation
        try:
            shutil.copy(self.path, target_path)
            print(f"Data copied to {target_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to copy data to {target_path}: {e}")

        return target_path

    def get_n_tr_ref(self):
        with h5py.File(self.path, "r") as f:
            n_tr_ref = f["n_tr"][:]

        return jnp.array(n_tr_ref)

    def get_fm_argdic(self):
        with h5py.File(self.path, "r") as f:
            argdic = json.loads(f.attrs["FM_ARGDIC"])

        return argdic


class SampleDataObj:

    def __init__(self, dir_obj, config):

        self.config = config
        self.__dict__.update(vars(config))

        exp_dir = dir_obj.exp_dir

        subdict = {"energies": None, "accepts": None, "time": None, "ch_runtime": None}
        self.val_dict = {i: subdict.copy() for i in range(config.n_chains)}

        self.fig_savepath = os.path.join(exp_dir, f"figures/sample_series.png")
        self.stats_fig_savepath = os.path.join(exp_dir, f"figures/stats.png")
        self.data_savepath = os.path.join(exp_dir, f"stats_00.hdf5")

    def set_vals(self, chain_idx, energies, time_, accepts, lf_steps, ch_runtime):

        self.val_dict[chain_idx]["energies"] = energies
        self.val_dict[chain_idx]["times"] = time_
        self.val_dict[chain_idx]["accepts"] = accepts
        self.val_dict[chain_idx]["lf_steps"] = lf_steps
        self.val_dict[chain_idx]["ch_runtime"] = ch_runtime

    def save_vals(self):

        total_runtime = sum(v["ch_runtime"] for v in self.val_dict.values())
        total_runtime_str = self.get_time_str(total_runtime)

        with h5py.File(self.data_savepath, "w") as f:
            h_g = f.create_group("Header")

            h_g.attrs["n_chains"] = self.n_chains
            h_g.attrs["n_epochs"] = self.n_epochs
            h_g.attrs["total_runtime"] = total_runtime_str

            f.create_dataset("samples", data=self.samples)
            f.create_dataset("temperatures", data=self.temperature)
            f.create_dataset("lf_steps", data=self.lf_steps)
            f.create_dataset("lf_step_min", data=self.lf_step_max)
            f.create_dataset("lf_step_max", data=self.lf_step_min)

            for i in range(self.n_chains):
                fg = f.create_group(f"ch_{i:02d}")

                fg.create_dataset("energies", data=self.val_dict[i]["energies"])
                fg.create_dataset("accepts", data=self.val_dict[i]["accepts"])
                fg.create_dataset("times", data=self.val_dict[i]["times"])
                fg.create_dataset("lf_steps", data=self.val_dict[i]["lf_steps"])
                fg.create_dataset("ch_runtime", data=self.val_dict[i]["ch_runtime"])

            n_chains = self.n_chains
            n_epochs = self.n_epochs
            boundaries = np.cumsum(self.samples)
            boundaries = np.insert(boundaries, 0, 0)

            grid_times = np.zeros((n_chains, n_epochs))
            grid_energies = np.zeros((n_chains, n_epochs))
            grid_accepts = np.zeros((n_chains, n_epochs))

            for i in range(n_chains):
                times = self.val_dict[i]["times"]
                energies = self.val_dict[i]["energies"]
                accepts = self.val_dict[i]["accepts"]
                for j in range(n_epochs):
                    start, end = boundaries[j], boundaries[j + 1]

                    grid_times[i, j] = np.mean(times[start:end])
                    grid_energies[i, j] = np.mean(energies[end - 1] - energies[start])
                    grid_accepts[i, j] = np.mean(accepts[start:end])

            # Store statistics in HDF5 file
            f.create_dataset("grid_times", data=grid_times)
            f.create_dataset("grid_energy_diff", data=grid_energies)
            f.create_dataset("grid_acc_ratio", data=grid_accepts)

    def plot(self, target_energy=None):

        figsize = 5
        ratio = 2.25
        ts = 8
        # fig, axs = plt.subplots(2, 1, figsize=(figsize * ratio, figsize))
        # plt.subplots_adjust(wspace=0.15, hspace=0.0)

        fig = plt.figure(figsize=(figsize * ratio, figsize))
        gs = GridSpec(
            2, 1, height_ratios=[2, 1]
        )  # First subplot twice the height of the second
        axs = [fig.add_subplot(gs[i]) for i in range(2)]
        plt.subplots_adjust(wspace=0.15, hspace=0.0)

        cmap = plt.get_cmap("nipy_spectral")
        colors = [
            cmap(i) for i in np.linspace(0.1, 0.9, self.config.n_chains, endpoint=True)
        ]

        n_energy_samples_mean = 10
        dy = 0.06

        max_M = 0
        for i, dic_key in enumerate(self.val_dict):

            energies = self.val_dict[dic_key]["energies"]

            if energies is None:
                continue

            accepts = jnp.array(self.val_dict[dic_key]["accepts"])
            acc_rates = jnp.cumsum(accepts) / jnp.arange(1, len(accepts) + 1)

            M = len(energies)
            max_M = max(M, max_M)
            x = jnp.arange(M)
            if i == 0:
                axs[0].set_xlim(x.min(), x.max())
                axs[1].set_xlim(x.min(), x.max())

            axs[0].scatter(x, energies, lw=0, s=6, color=colors[i])
            axs[1].plot(x, acc_rates, color=colors[i])

            if M > n_energy_samples_mean:

                if target_energy is not None:
                    dif = (
                        np.mean(np.array(energies)[-n_energy_samples_mean:]) - target_energy
                    )
                    energy_str = fr"$\Delta$H: {dif:.3e}"
                else:
                    energy_str = f"H: {energies[-1]:.3e}"

                axs[0].text(
                    1.01,
                    1 - dy * i,
                    energy_str,
                    transform=axs[0].transAxes,
                    fontsize=ts,
                    va='top',
                    color = colors[i]
                )

        yl = axs[0].get_ylim()
        boundaries = np.cumsum(self.samples)
        boundaries = np.insert(boundaries, 0, 0)

        if target_energy is not None:
            axs[0].axhline(target_energy, ls="--", c="grey")
            target_energy_str = f"H*: {target_energy:.3e}"
            axs[0].text(
                1, target_energy, target_energy_str, va="bottom", ha="left", fontsize=ts
            )

        for i, x in enumerate(boundaries[:-1]):
            if x < max_M:
                axs[0].axvline(x, ls="--", c="grey", alpha=0.5)

                T = self.temperature[i]
                text = f"$T={T**(1/3):0.0f}^3$\n"
                axs[0].text(
                    boundaries[i] + 1, yl[1], text, ha="left", va="top", fontsize=ts
                )

        axs[0].set_ylabel("Energy")
        axs[1].set_ylabel("acc rate")
        axs[1].set_xlabel("Sample index")

        axs[1].set_ylim(0, 1)

        axs[0].tick_params(labelbottom=False)

        for ax in axs:
            ax.grid(
                which="major",
                linewidth=0.5,
                color="k",
                alpha=0.25,
            )

        fig.savefig(self.fig_savepath, bbox_inches="tight")
        plt.close()

    def plot_statistics_tables(self, runtime=None):
        with h5py.File(self.data_savepath, "r") as f:
            grids = {
                "Mean Times": f["grid_times"][:],
                "Energy Differences": f["grid_energy_diff"][:],
                "Acceptance Ratios": f["grid_acc_ratio"][:],
            }

            n_chains, n_epochs = grids["Mean Times"].shape

            # Retrieve additional metadata
            samples = f["samples"][:]
            temperatures = f["temperatures"][:]
            lf_steps = f["lf_steps"][:]
            lf_step_min = f["lf_step_min"][:]
            lf_step_max = f["lf_step_max"][:]

            # Create plot
            fs, ratio = 10, 1.2
            ts = 15
            ts_ = ts * 1
            lw = 0.3
            vs = 0.19
            y0 = 1.1
            y = 0.95
            font = "monospace"
            fig, axes = plt.subplots(4, 1, figsize=(fs * ratio, fs))

            temperatures = temperatures ** (1 / 3)
            metadata = np.array(
                [samples, temperatures, lf_steps, lf_step_min, lf_step_max]
            )
            metadata_labels = [
                "N_SAMP",
                "TEMP*",
                "N_LF",
                "LF_MIN",
                "LF_MAX",
            ]

            formats = ["{:.0f}", "{:.1f}", "{:.0f}", "{:.2e}", "{:.2e}"]
            table_data = [
                [formats[i].format(metadata[i, j]) for j in range(n_epochs)]
                for i in range(len(metadata))
            ]
            col_labels = [f"EP{i}" for i in range(n_epochs)]
            row_labels = metadata_labels

            axes[0].axis("off")
            metadata_table = axes[0].table(
                cellText=table_data,
                rowLabels=row_labels,
                colLabels=col_labels,
                cellLoc="center",
                loc="center",
            )
            metadata_table.auto_set_font_size(False)
            metadata_table.set_fontsize(ts_)
            axes[0].set_title(
                "Metadata", fontsize=ts, y=y0, fontfamily=font, color="white"
            )

            formats = ["{:.02e}", "{:.2e}", "{:.03f}"]
            for idx, (title, grid) in enumerate(grids.items()):
                idx += 1
                grid_with_means = np.zeros((n_chains + 1, n_epochs + 1))
                grid_with_means[:-1, :-1] = grid
                grid_with_means[:-1, -1] = np.mean(grid, axis=1)  # Mean per chain
                grid_with_means[-1, :-1] = np.mean(grid, axis=0)  # Mean per epoch
                grid_with_means[-1, -1] = np.mean(grid)  # Overall mean

                table_data = [
                    [
                        formats[idx - 1].format(grid_with_means[i, j])
                        for j in range(n_epochs + 1)
                    ]
                    for i in range(n_chains + 1)
                ]
                col_labels = [f"EP{i}" for i in range(n_epochs)] + [""]
                row_labels = [f"CH{i}" for i in range(n_chains)] + [""]

                axes[idx].axis("off")
                table = axes[idx].table(
                    cellText=table_data,
                    rowLabels=row_labels,
                    colLabels=col_labels,
                    cellLoc="center",
                    loc="center",
                )
                table.auto_set_font_size(False)
                table.set_fontsize(ts_)
                axes[idx].set_title(
                    title, fontsize=ts, y=y, fontfamily=font, color="white"
                )

                for key, cell in table.get_celld().items():
                    cell.set_text_props(color="white")
                    cell.set_facecolor("black")
                    cell.set_edgecolor("white")
                    cell.set_linewidth(lw)
                    cell.set_text_props(fontfamily=font)
                    cell.set_height(vs)

            for key, cell in metadata_table.get_celld().items():
                cell.set_text_props(color="white")
                cell.set_facecolor("black")
                cell.set_edgecolor("white")
                cell.set_linewidth(lw)
                cell.set_text_props(fontfamily=font)
                cell.set_height(vs)

            if runtime is not None:
                # add text to fig
                fig.text(
                    0.1,
                    0.0,
                    "Runtime: {}".format(self.get_time_str(runtime)),
                    ha="center",
                    va="bottom",
                    fontsize=ts,
                    fontfamily=font,
                    color="grey",
                )

            fig.patch.set_facecolor("black")
            plt.tight_layout(rect=[0, 0, 1, 1])
            plt.savefig(self.stats_fig_savepath)
            plt.close()

    @staticmethod
    def get_time_str(time):
        hours, remainder = divmod(time, 3600)
        minutes, seconds = divmod(remainder, 60)

        parts = []
        if hours > 0:
            parts.append(f"{int(hours)}h")
        if minutes > 0:
            parts.append(f"{int(minutes)}m")

        parts.append(f"{seconds:.4f}s")

        return " ".join(parts)
