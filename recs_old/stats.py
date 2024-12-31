import os
import pandas as pd
import h5py
import numpy as np


def save_statistics_to_spreadsheets(exp_dir):
    stats_datapath = os.path.join(exp_dir, "stats_00.hdf5")
    stats_folder = os.path.join(exp_dir, "stats")
    os.makedirs(stats_folder, exist_ok=True)

    with h5py.File(stats_datapath, "r") as f:
        grids = {
            "mean_times": f["grid_times"][:],
            "energy_diffs": f["grid_energy_diff"][:],
            "acc_ratios": f["grid_acc_ratio"][:],
        }

        n_chains, n_epochs = grids["mean_times"].shape

        metadata = np.array(
            [
                f["samples"][:],
                f["temperatures"][:],
                f["lf_steps"][:],
                f["lf_step_min"][:],
                f["lf_step_max"][:],
            ]
        )
        metadata_labels = [
            "N_SAMP",
            "TEMP",
            "N_LF",
            "LF_MIN",
            "LF_MAX",
        ]

        # Save metadata to a spreadsheet
        metadata_df = pd.DataFrame(metadata.T, columns=metadata_labels)
        metadata_df.to_excel(os.path.join(stats_folder, "metadata.xlsx"), index=False)

        # Save each grid to a separate spreadsheet
        for name, grid in grids.items():
            grid_with_means = np.zeros((n_chains + 1, n_epochs + 1))
            grid_with_means[:-1, :-1] = grid
            grid_with_means[:-1, -1] = np.mean(grid, axis=1)  # Mean per chain
            grid_with_means[-1, :-1] = np.mean(grid, axis=0)  # Mean per epoch
            grid_with_means[-1, -1] = np.mean(grid)  # Overall mean

            col_labels = [f"EP{i}" for i in range(n_epochs)] + ["Mean"]
            row_labels = [f"CH{i}" for i in range(n_chains)] + ["Mean"]

            df = pd.DataFrame(grid_with_means, index=row_labels, columns=col_labels)
            df.to_excel(os.path.join(stats_folder, f"{name}.xlsx"))
