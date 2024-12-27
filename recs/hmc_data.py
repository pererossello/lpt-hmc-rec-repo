import h5py
import os
import shutil
import json

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
