# python -m recs.main

import os
import sys
import inspect
import shutil
from functools import partial
import json
import time

import matplotlib.pyplot as plt
import h5py
import jax
import jax.numpy as jnp
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from hmc.main import hmc_sample, hmc_sample_epochs

from recs.hmc_config import HMCRootConfig
from recs.hmc_aux_config import HMCAuxConfig, IdxClass
from recs.hmc_data import HMCData
from recs.dir_management import DirectoryManager
from recs.plot_utils import HMCPlot
from recs.hmc_setup import HMCSetUp


class HMCRootSampler:
    def __init__(
        self,
        rec_dir_name,
        exp_dir_name,
        config: HMCRootConfig,
        aux_config: HMCAuxConfig,
        fold_path=None,
    ):
        self.config = config
        self.__dict__.update(vars(config))

        self.aux_config = aux_config
        self.__dict__.update(vars(aux_config))

        self.dir_manager = DirectoryManager(base_dir=rec_dir_name, fold_path=fold_path)
        proceed = self.dir_manager.create_exp_dir(exp_dir_name)
        if not proceed:
            raise RuntimeError(
                f"Experiment directory '{exp_dir_name}' creation failed. "
                "Sampling aborted."
            )
        self.dir_manager.create_root_chain_subdirs(config.n_chains)

        self.config.save_to_json(self.dir_manager.exp_dir)

        self.dir_manager.create_figures_dir(config.n_chains)

    def postprocess():

        return

    def run_hmc(self):

        # plotting
        self.fig_obj = HMCPlot(
            self.dir_manager, self.config, self.aux_config, self.argdic
        )
        self.fig_obj.plot_reference()

        # data_obj
        my_data_obj = SampleDataObj(self.dir_manager, self.config)

        idx_obj = IdxClass(self.config, self.aux_config)
        idx_obj.print_initial_message()

        key_sample = jax.random.PRNGKey(self.config.seed_int_chain)
        skeys_chains = jax.random.split(key_sample, self.n_chains)

        ti_0 = time.perf_counter()
        for i in range(self.n_chains):

            energies, times, accs = [], [], []

            idx_obj.plot_idx = 0
            idx_obj.chain_sample_idx = 0
            idx_obj.chain_idx = i
            idx_obj.print_current_chain()

            skeys_epochs = jax.random.split(skeys_chains[i], self.n_epochs)

            q = self._read_initial_states(i)
            self._save_sample(q, i, first=True)

            for j in range(self.n_epochs):
                idx_obj.epoch_sample_idx = 0
                idx_obj.epoch_idx = j
                idx_obj.print_current_epoch()

                skeys_samples = jax.random.split(skeys_epochs[j], self.samples[j])

                hmc_sample_ = partial(
                    hmc_sample_epochs,
                    sample_momenta=self.sample_p,
                    leapfrog=self.leapfrog,
                    metropolis=self.metropolis,
                    lf_steps_min=jnp.array(self.lf_step_min),
                    lf_steps_max=jnp.array(self.lf_step_max),
                    lf_steps=jnp.array(self.lf_steps),
                    temperatures=jnp.array(self.temperature),
                )

                hmc_sample_ = jax.jit(hmc_sample_)

                # run to compile
                _ = hmc_sample_(q, key_sample, 0)

                for k in range(self.samples[j]):
                    idx_obj.epoch_sample_idx = k

                    # sample q
                    ti_1 = time.perf_counter()
                    q, accept, alpha, ham, delta_h, lf_step = hmc_sample_(
                        q, skeys_samples[k], j
                    )
                    tf_1 = time.perf_counter()

                    # get acceptance rate
                    idx_obj.add_accept(accept)
                    acc_rate = idx_obj.get_acc_rat()

                    # fill
                    energies.append(ham)
                    times.append(tf_1- ti_1)
                    accs.append(accept)

                    # Handle saving
                    do_save = self.aux_config.should_save(idx_obj)
                    if do_save:
                        self._save_sample(q, i, first=False)

                    # Handle plotting
                    do_plot = self.aux_config.should_plot(idx_obj)
                    if do_plot:
                        self.fig_obj.plot_sample(q, idx_obj)
                        idx_obj.plot_idx += 1

                        my_data_obj.set_vals(i, energies, times, accs)
                        
                        if idx_obj.total_sample_idx>0:
                            my_data_obj.plot(idx_obj)

                    # Handle printing
                    do_print = self.aux_config.should_print(idx_obj.chain_sample_idx)
                    if do_print:
                        idx_obj.print_idxs()
                        idx_obj.print_hmc(
                            ham, delta_h, alpha, accept, acc_rate, lf_step
                        )
                        idx_obj.print_times()
                        print("\n")

                    tf_2 = time.perf_counter()
                    # Handle timing
                    idx_obj.runtime = tf_1 - ti_0
                    idx_obj.cum_sample_time += tf_1 - ti_1
                    idx_obj.cum_loop_time += tf_2 - ti_1

                    idx_obj.chain_sample_idx += 1
                    idx_obj.total_sample_idx += 1

        my_data_obj.save_vals()
        my_data_obj.plot_statistics_tables()

    def set_up_hmc(self):

        n_tr_ref = self.data_obj.get_n_tr_ref()

        self.n = self.data_obj.n
        self.l = self.data_obj.l
        z_i = self.data_obj.z_i

        self.hmc_setup = HMCSetUp(
            self.config, n_tr_ref, self.argdic, self.n, self.l, z_i
        )

        # args: q, p, lf_step_min, lf_step_max, lf_steps, key
        self.leapfrog = self.hmc_setup.get_leapfrog()
        self.metropolis = self.hmc_setup.get_metropolis()  # args: key, hamiltonian
        self.hamiltonian = self.hmc_setup.get_hamiltonian()  # args: q, p
        self.sample_p = self.hmc_setup.get_sample_momenta()  # args: key

    def set_data(self, path):

        self.data_obj = HMCData(path)
        self.data_obj.copy_to(self.dir_manager.working_directory, "data_ref")

        if self.data_obj.inverse_crime:
            print("Inverse Crime: reading argdic from datapath.")
            self.argdic = self.data_obj.get_fm_argdic()
            self.argdic["SAMPLE"] = False
        else:
            print("Not Inverse Crime. Please provide Forward Model argdic")

    def set_fm_argdic(self, argdic):
        if argdic is not None:
            self.argdic = argdic

        savepath = os.path.join(self.dir_manager.working_directory, "fm_config.json")
        with open(savepath, "w") as json_file:
            json.dump(self.argdic, json_file, indent=4)

    def read_or_make_initial_states(self):

        is_kind = self.config.initial_state

        if is_kind == "RANDOM":
            key_is = jax.random.PRNGKey(self.config.seed_int_is)
            skeys_is = jax.random.split(key_is, self.config.n_chains)

            initial_states = self.hmc_setup.make_initial_states(skeys_is)

        elif isinstance(is_kind, list):
            raise ValueError("Reading ISS Not Implemented Yet!")

        self._save_initial_states(initial_states)

    def _save_initial_states(self, initial_states):

        self.iss_path = os.path.join(self.dir_manager.exp_dir, "initial_states.hdf5")
        with h5py.File(self.iss_path, "w") as f:

            header = f.create_group("Header")
            header.attrs["N"] = self.argdic["N"]
            header.attrs["L"] = self.argdic["L"]
            header.attrs["IC_KIND"] = self.argdic["IC_KIND"]

            for i, is_ in enumerate(initial_states):
                f.create_dataset(f"is_{i:02d}", data=is_)

    def _read_initial_states(self, i):
        with h5py.File(self.iss_path, "r") as f:
            is_ = f[f"is_{i:02d}"][:]
        return is_

    def _save_sample(self, data, chain_idx, first=False):

        savedir = self.dir_manager.root_dirs[chain_idx]
        savepath = os.path.join(savedir, "samples.hdf5")

        if first:
            with h5py.File(savepath, "w") as f:

                header = f.create_group("Header")

                header.attrs["N"] = self.n
                header.attrs["L"] = self.l
                header.attrs["next_idx"] = 0  # Initialize counter

                smpl_group = f.create_group("samples_00")
                smpl_group.create_dataset(f"{0:04d}", data=data)

        else:
            with h5py.File(savepath, "a") as f:
                next_idx = f["Header"].attrs["next_idx"] + 1

                f["samples_00"].create_dataset(f"{next_idx:04d}", data=data)

                f["Header"].attrs["next_idx"] = next_idx


class SampleDataObj:

    def __init__(self, dir_obj, config):

        self.config = config
        self.__dict__.update(vars(config))

        exp_dir = dir_obj.exp_dir

        subdict = {"energies": None, "accepts": None, 'time': None}
        self.val_dict = {i: subdict.copy() for i in range(config.n_chains)}

        self.fig_savepath = os.path.join(exp_dir, f"figures/sample_series.png")
        self.stats_fig_savepath = os.path.join(exp_dir, f"figures/stats.png")
        self.data_savepath = os.path.join(exp_dir, f"stats_00.hdf5")

    def set_vals(self, chain_idx, energies, time_, accepts):

        self.val_dict[chain_idx]["energies"] = energies
        self.val_dict[chain_idx]["times"] = time_
        self.val_dict[chain_idx]["accepts"] = accepts

    def save_vals(self):

        with h5py.File(self.data_savepath, "w") as f:
            h_g = f.create_group("Header")

            h_g.attrs["n_chains"] = self.n_chains
            h_g.attrs["n_epochs"] = self.n_epochs

            f.create_dataset("samples", data=self.samples)
            f.create_dataset("temperatures", data=self.temperature)
            f.create_dataset("lf_steps", data=self.lf_steps)
            f.create_dataset("lf_step_min", data=self.lf_step_max)
            f.create_dataset("lf_step_max", data=self.lf_step_min)

            for i in range(self.n_chains):
                fg = f.create_group(f'ch_{i:02d}')

                fg.create_dataset('energies', data=self.val_dict[i]['energies'])
                fg.create_dataset('accepts', data=self.val_dict[i]['accepts'])
                fg.create_dataset('times', data=self.val_dict[i]['times'])

            n_chains = self.n_chains
            n_epochs = self.n_epochs
            boundaries = np.cumsum(self.samples)
            boundaries = np.insert(boundaries, 0, 0) 

            grid_times = np.zeros((n_chains, n_epochs))
            grid_energies = np.zeros((n_chains, n_epochs))
            grid_accepts = np.zeros((n_chains, n_epochs))

            for i in range(n_chains):
                times = self.val_dict[i]['times']
                energies = self.val_dict[i]['energies']
                accepts = self.val_dict[i]['accepts']
                for j in range(n_epochs):
                    start, end = boundaries[j], boundaries[j + 1]
                    
                    grid_times[i, j] = np.mean(times[start:end])
                    grid_energies[i, j] = np.mean(energies[end-1]-energies[start])
                    grid_accepts[i, j] = np.mean(accepts[start:end])

            # Store statistics in HDF5 file
            f.create_dataset('grid_times', data=grid_times)
            f.create_dataset('grid_energy_diff', data=grid_energies)
            f.create_dataset('grid_acc_ratio', data=grid_accepts)



    def plot(self, idx_obj):

        figsize = 5
        ratio = 2.25
        fig, axs = plt.subplots(1, 1, figsize=(figsize * ratio, figsize))
        plt.subplots_adjust(wspace=0.15, hspace=0.0)

        cmap = plt.get_cmap("rainbow")
        colors = [
            cmap(i) for i in np.linspace(0.1, 0.9, self.config.n_chains, endpoint=True)
        ]

        max_M = 0
        for i, dic_key in enumerate(self.val_dict):

            energies = self.val_dict[dic_key]["energies"]

            if energies is None:
                continue

            M = len(energies)
            max_M = max(M, max_M)
            x = jnp.arange(M)
            if i == 0:
                axs.set_xlim(x.min(), x.max())

            axs.scatter(
                x, energies, lw=0, s=6, color=colors[i]
            )  

        yl = axs.get_ylim()
        boundaries = np.cumsum(self.samples)
        boundaries = np.insert(boundaries, 0, 0) 

        
        for i, x in enumerate(boundaries[:-1]):
            if x < max_M:
                axs.plot([x, x], [yl[0], yl[1]], ls="--", c="k", alpha=0.5)

                T = self.temperature[i]
                text = f'T={T**(1/3):0.1f}^3'
                axs.text(boundaries[i]+1, yl[1], text ,ha='left', va='center')

        
        # T = self.temperature[0]
        # text = f'T={T**(1/3):0.1f}^3'
        # axs.text(1, yl[1], text ,ha='left', va='center')

        # if max_M > temp_change_idxs[-2]:
        #     T = self.temperature[-1]
        #     text = f'T={T**(1/3):0.1f}^3'
        #     axs.text(temp_change_idxs[-2]+1, yl[1], text ,ha='left', va='center')

        axs.set_ylabel('Energy')
        axs.set_xlabel('Sample index')

        fig.savefig(self.fig_savepath, bbox_inches="tight")


    def plot_statistics_tables(self):
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
            ts_ = ts*1
            lw = 0.3
            vs = 0.19
            y0=1.1
            y = 0.95
            font = 'monospace'
            fig, axes = plt.subplots(4, 1, figsize=(fs * ratio, fs))

            temperatures = temperatures**(1/3)
            metadata = np.array([samples, temperatures, lf_steps, lf_step_min, lf_step_max])
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
            axes[0].set_title("Metadata", fontsize=ts, y=y0, fontfamily=font)


            formats = ["{:.02e}", "{:.2e}", "{:.03f}"]
            for idx, (title, grid) in enumerate(grids.items()):
                idx += 1
                grid_with_means = np.zeros((n_chains + 1, n_epochs + 1))
                grid_with_means[:-1, :-1] = grid
                grid_with_means[:-1, -1] = np.mean(grid, axis=1)  # Mean per chain
                grid_with_means[-1, :-1] = np.mean(grid, axis=0)  # Mean per epoch
                grid_with_means[-1, -1] = np.mean(grid)  # Overall mean

                table_data = [
                    [formats[idx-1].format(grid_with_means[i, j]) for j in range(n_epochs + 1)]
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
                axes[idx].set_title(title, fontsize=ts, y=y, fontfamily=font)


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

            plt.style.use('dark_background')
            plt.tight_layout(rect=[0, 0, 1, 1])
            plt.savefig(self.stats_fig_savepath)
            plt.close()