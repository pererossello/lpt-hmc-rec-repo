import os
import json

import jax
import jax.numpy as jnp

from recs.hmc_config import HMCRootConfig


class HMCAuxConfig:
    _presets = {
        "default": {
            "save_n_first": 0,
            "save_n_last": 50,
            "save_last_per_epoch": False,
            "save_n_first_per_epoch": 0,
            "save_n_last_per_epoch": 0,
            "print_step": 5,
            "plot_first": True,
            "plot_last": True,
            "plot_step": 50,
            "n_bins": 20,
        },
    }

    def __init__(self, preset=None, **kwargs):
        config = self._presets.get(preset, {}).copy()
        config.update(kwargs)

        # Direct assignment with defaults
        self.save_n_first = config.get("save_n_first", 0)
        self.save_n_last = config.get("save_n_last", 0)
        self.save_last_per_epoch = config.get("save_last_per_epoch", False)
        self.save_n_first_per_epoch = config.get("save_n_first_per_epoch", 0)
        self.save_n_last_per_epoch = config.get("save_n_last_per_epoch", 0)

        self.print_step = config.get("print_step", 5)

        self.plot_first = config.get("plot_first", True)
        self.plot_last = config.get("plot_last", True)
        self.plot_step = config.get("plot_step", 100)
        self.n_bins = config.get("n_bins", 20)

    def should_save(self, idx_obj):
        # Save first N global samples
        if idx_obj.chain_sample_idx < self.save_n_first:
            return True

        # Save last N samples of chain
        if idx_obj.chain_sample_idx >= idx_obj.samples_per_chain - self.save_n_last:
            return True

        # Save first N samples of each epoch
        if idx_obj.epoch_sample_idx < self.save_n_first_per_epoch:
            return True

        # Save last N samples of each epoch
        if idx_obj.epoch_sample_idx >= max(
            0,
            idx_obj.samples_per_chain // (idx_obj.epoch_idx + 1)
            - self.save_n_last_per_epoch,
        ):
            return True

        # Save the last sample of each epoch
        if (
            self.save_last_per_epoch
            and idx_obj.epoch_sample_idx == idx_obj.samples_per_epoch - 1
        ):
            return True

        return False

    def should_print(self, global_k):
        return global_k % self.print_step == 0

    def should_plot(self, idx_obj):
        # Plot first sample
        if idx_obj.chain_sample_idx == 0 and self.plot_first:
            return True

        # Plot last sample
        if idx_obj.chain_sample_idx == idx_obj.samples_per_chain - 1 and self.plot_last:
            return True

        # Plot every N steps
        if idx_obj.chain_sample_idx % self.plot_step == 0:
            return True

        return False


class IdxClass:

    def __init__(self, config: HMCRootConfig, aux_config: HMCAuxConfig):

        self.__dict__.update(vars(config))
        self.__dict__.update(vars(aux_config))

        self.chain_idx = 0
        self.epoch_idx = 0
        self.epoch_sample_idx = 0
        self.chain_sample_idx = 0
        self.total_sample_idx = 0
        self.plot_idx = 0

        self.runtime = 0
        self.cum_sample_time = 0
        self.cum_loop_time = 0

        self.accepts = [0] * self.n_chains
        self.accepts_dic = {i: {j: 0 for j in range(self.n_epochs)} for i in range(self.n_chains)}

    def add_accept(self, accept):
        self.accepts[self.chain_idx] += int(accept)
        self.accepts_dic[self.chain_idx][self.epoch_idx] += int(accept)

    def get_acc_rat(self):
        self.acc_ratio = self.accepts[self.chain_idx] / (self.chain_sample_idx + 1)
        self.acc_ratio_dic = {}
        for i in range(self.chain_idx+1):
            self.acc_ratio_dic[i] = {}
            for j in range(self.epoch_idx+1):
                val = (self.epoch_sample_idx+1) if j==self.epoch_idx else self.samples[j] 
                self.acc_ratio_dic[i][j] = self.accepts_dic[i][j]/ val

        return self.acc_ratio

    def print_initial_message(self):
        print(
            "\n"
            f"Sampling {self.n_chains} chains\n"
            f"for {self.n_epochs} epochs: {self.samples}\n"
            f"for a total of {self.samples_per_chain} samples per chain.\n"
        )

    def print_current_chain(self):
        print(f"\nSampling CHAIN {self.chain_idx+1:02d}/{self.n_chains:02d}\n")

    def print_current_epoch(self):
        print(f"Starting EPOCH {self.epoch_idx+1:02d}/{self.n_epochs:02d}\n")

    def print_idxs(self):
        print(
            f"CH: {self.chain_idx+1}/{self.n_chains} | "
            f"EP: {self.epoch_idx+1}/{self.n_epochs} | "
            f"i: {self.epoch_sample_idx:04d}/{self.samples[self.epoch_idx]:04d} | "
            f"I: {self.chain_sample_idx:04d}/{self.samples_per_chain:04d}"
        )

    def print_times(self):
        mtps = self.cum_sample_time / (self.total_sample_idx + 1)
        mtpl = self.cum_loop_time / (self.total_sample_idx + 1)
        ettt = mtpl * self.total_samples
        ette = mtpl * (self.total_samples - self.total_sample_idx + 1)
        print(
            f"MTPS: {self.get_time_str(mtps)} | "  # mean time per sample
            f"MTPL: {self.get_time_str(mtpl)} | "   # mean time per loop
            f"RUNTIME: {self.get_time_str(self.runtime)} \n"  # runtime
            f"EXPECTED_TIME: {self.get_time_str(ettt)} \n"
            f"EXPECTED_TIME_LEFT: {self.get_time_str(ette)}"  # expected time to end
        )

    def print_hmc(self, ham, delta_h, target_energy, accept, acc_rate, lf_step):

        dh = ham-target_energy

        print(f"H: {ham:04e} | " f"DH: {delta_h:04e} | " f"H-H*: {dh:04e} | ")
        print(
            f"ACC_RAT: {acc_rate:03f} | " f"ACC: {accept} | " f"LF_STEP: {lf_step:04e} | "
        )
        print(f"T: {(self.temperature[self.epoch_idx])**(1/3):0.1f}^3")

        for i in range(self.epoch_idx+1):
            print(f"ACC_RAT_{i}: {self.acc_ratio_dic[self.chain_idx][i]:0.3f} |", end=' ')
        print('')

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
