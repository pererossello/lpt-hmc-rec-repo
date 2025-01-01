import os
import shutil
import json

import jax.numpy as jnp
import h5py 

from .grove import Grove

class HMCSampler:

    def __init__(self, forest_name, grove_name, base_dir):

        self.forest_dir = os.path.join(base_dir, forest_name)
        self.grove_dir = os.path.join(self.forest_dir, grove_name)

        self.ChGr = Grove(self.grove_dir)

        self.handle_data() # sets DataObj as attr

        return
    

    """
    Methods to start chains (trees/trunks)

    """

    def plant_trees(self,):


        return 



    """
    Methods to deal with reference data
    """

    def handle_data(self):

        self.data_is_set = False

        self.data_path = os.path.join(self.forest_dir, 'data.hdf5')
        if os.path.exists(self.data_path):
            self.DataObj = DataHelper(self.data_path)
            self.data_is_set = True
            self._copy_some_attrs()
        else:
            print("No data in forest. Provide path with '.set_data()'")
            
        return 
    def set_data(self, data_path):
        self.DataObj = DataHelper(data_path)
        self.DataObj.copy_to(self.forest_dir, 'data')
        self.data_is_set = True
        self._copy_some_attrs()

    def _copy_some_attrs(self):
        self.n = self.DataObj.n
        self.l = self.DataObj.l
        self.z_i = self.DataObj.z_i


class DataHelper():

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