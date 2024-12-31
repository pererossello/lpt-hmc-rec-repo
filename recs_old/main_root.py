# python -m recs.main

import os
import sys
import inspect
import shutil
from functools import partial
import json
import time

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import h5py
import jax
import jax.numpy as jnp
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from hmc.main import hmc_sample, hmc_sample_epochs

from recs.hmc_config import HMCRootConfig
from recs.hmc_aux_config import HMCAuxConfig, IdxClass
from recs.hmc_data import HMCData, SampleDataObj
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
            ti_ch = time.perf_counter()

            energies, times, accs, lf_steps = [], [], [], []

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
                    times.append(tf_1 - ti_1)
                    accs.append(accept)
                    lf_steps.append(lf_step)

                    # Handle saving
                    do_save = self.aux_config.should_save(idx_obj)
                    if do_save:
                        self._save_sample(q, i, first=False)

                    # Handle plotting
                    do_plot = self.aux_config.should_plot(idx_obj)
                    if do_plot:
                        self.fig_obj.plot_sample(q, idx_obj)
                        idx_obj.plot_idx += 1

                        ch_runtime = tf_1 - ti_ch 
                        my_data_obj.set_vals(i, energies, times, accs, lf_steps, ch_runtime)

                        if idx_obj.total_sample_idx > 0:
                            my_data_obj.plot(self.data_obj.hmc_target_energy)

                    # Handle printing
                    do_print = self.aux_config.should_print(idx_obj.chain_sample_idx)
                    if do_print:
                        idx_obj.print_idxs()
                        idx_obj.print_hmc(
                            ham, delta_h, self.data_obj.hmc_target_energy, accept, acc_rate, lf_step
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
        plt.close()

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

