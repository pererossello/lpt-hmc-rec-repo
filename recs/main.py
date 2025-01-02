import os
import sys
import shutil
import json
from functools import partial

import jax
import jax.numpy as jnp
import h5py

from .grove import Grove

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from forward_model.utils import compute_or_load_pow_spec
from forward_model.energies import make_is_u, make_is_delta


class HMCSampler:

    def __init__(self, forest_name, grove_name, base_dir):

        self.forest_dir = os.path.join(base_dir, forest_name)
        self.grove_dir = os.path.join(self.forest_dir, grove_name)

        self.ChGr = Grove(self.grove_dir)

        self.handle_data()  # sets DataObj as attr

        return

    def clear_grove(self):

        self.ChGr.clear_grove()
        self.ChGr = Grove(self.grove_dir)


    """
    Methods to set up the HMC sampling

    COMMENT: If leapfrog method and forward_model are different for different chains, I think i will have to define different different functions and compile them at the start of each chain, and remove the previoius ones to delete them (as they contain arrays in the definition and may cause memory to define/load and compile them beforehand)

    I NEED: A function that given a node, returns the sample function, and then i compile when i want. 
    """


    def run_hmc():

        # Some checks that everything is set up for sampling would be great

        return











    """
    Methods to start chains (trees/trunks)
    """

    def plant_trees(self, init_state_input, hmc_config_list):
        """
        :init_state_input: int or list of ints or paths
        :hmc_config: dic
        """

        self._handle_init_state_input(init_state_input)
        self.nodes = self.ChGr.add_n_trunks(self.n_nodes)

        if not isinstance(hmc_config_list, list):
            hmc_config_list = [hmc_config_list] * self.n_trees

        for i, node in enumerate(self.nodes):
            node.hmc_config = hmc_config_list[i].to_dict()

        self._read_or_make_trunk_init_states()

        return

    def _read_or_make_trunk_init_states(self):

        for i, init_state_spec in enumerate(self.init_state_spec_list):
            if isinstance(init_state_spec, str):
                raise NotImplementedError(
                    f"String path handling is not implemented yet: {init_state_spec}"
                )

            else:
                if self.ic_kind in ["FSK_U", "U"]:
                    q_init = make_is_u(init_state_spec, self.n)
                elif self.ic_kind in ["DELTA", "DELTA_HAT"]:
                    q_init = make_is_delta(
                        init_state_spec, self.n, self.l, self.pow_spec
                    )

            path = os.path.join(self.nodes[i].path, "samples.hdf5")
            with h5py.File(path, "w") as f:
                h_g = f.create_group("Header")
                h_g.attrs['hmc_config'] = json.dumps(self.nodes[i].hmc_config)
                h_g.attrs['N'] = self.n
                h_g.attrs['L'] = self.l
                h_g.attrs['Z_I'] = self.z_i
                h_g.attrs["IC_KIND"] = self.ic_kind

                f.create_dataset("initial_state", data=q_init)

                f.create_group("Samples")

    def _handle_init_state_input(self, init_state_input):
        """
        Sets attr :init_state_spec_list: which is a list of either paths or keys
        """
        if isinstance(init_state_input, int):
            self.n_nodes = init_state_input
            key_ = jax.random.PRNGKey(1)
            self.init_state_spec_list = jax.random.split(key_, self.n_nodes)

        elif isinstance(init_state_input, list):
            self.n_nodes = len(init_state_input)
            self.init_state_spec_list = []
            for element in init_state_input:
                if isinstance(element, str):
                    self.init_state_spec_list.append(element)
                elif isinstance(element, int):
                    self.init_state_spec_list.append(jax.random.PRNGKey(element))
                else:
                    raise ValueError(
                        "All elements in init_state_input must be either integers or strings."
                    )
        else:
            raise ValueError(
                "init_state_input must be an integer or a list of integers or strings."
            )

    """
    Methods to deal with reference data
    """

    def handle_data(self):

        self.data_is_set = False

        self.data_path = os.path.join(self.forest_dir, "data.hdf5")
        if os.path.exists(self.data_path):
            self.DataObj = DataHelper(self.data_path)
            self.data_is_set = True
            self._copy_some_attrs()
            self._set_pow_spec()
        else:
            print("No data in forest. Provide path with '.set_data()'")

        return

    def set_data(self, data_path):
        self.DataObj = DataHelper(data_path)

        """
        Here is could implement that if data_ref already in the forest dir then I check if the hdf5 file i want to set is identical or not to the data.hdf5 and if not rise a warning and ask if I want to overwrite it. But this is a pretty niche problem, so no need to care about it now. 
        """
        # if self.data_is_set:
        #     confirmation = input(f"Data already in forest folder '{self.grove_name}'? [y/N]: ")

        self.data_is_set = True
        self.DataObj.copy_to(self.forest_dir, "data")
        self._copy_some_attrs()
        self._set_pow_spec()

    def _copy_some_attrs(self):
        self.ic_kind = self.DataObj.ic_kind
        self.n = self.DataObj.n
        self.l = self.DataObj.l
        self.z_i = self.DataObj.z_i

    def _set_pow_spec(self):
        if self.ic_kind in ["U", "FSK_U"]:
            self.pow_spec = None
            self.inv_pow_spec = None
        elif self.ic_kind in ["DELTA", "DELTA_HAT"]:
            self.pow_spec = compute_or_load_pow_spec(self.n, self.l, self.z_i)
            self.inv_pow_spec = jnp.where(self.pow_spec != 0.0, 1 / self.pow_spec, 0.0)


class OutputHelper:

    def __init__(self, node_dir):

        return


class DataHelper:

    def __init__(self, path):
        self.path = path
        self._load_header_attributes()

        if self.inverse_crime:
            self.fm_config = self.get_fm_config()

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

    def get_n_tr(self):
        with h5py.File(self.path, "r") as f:
            n_tr = f["n_tr"][:]

        return jnp.array(n_tr)

    def get_fm_config(self):
        with h5py.File(self.path, "r") as f:
            fm_config = json.loads(f.attrs["fm_config"])

        return fm_config
