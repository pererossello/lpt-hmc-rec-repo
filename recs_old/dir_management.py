import inspect
import os
import shutil


class DirectoryManager:
    def __init__(self, base_dir, fold_path):
        self.base_dir = base_dir

        if fold_path is None:
            self.calling_directory = self._get_calling_directory()
            self.working_directory = os.path.join(self.calling_directory, base_dir)
        else:
            self.working_directory = os.path.join(fold_path, self.base_dir)
        
        self.exp_dir = None  # Initialize as None to track if created

        self.root_chain_basename = 'root_chain'

    def _get_calling_directory(self):
        return os.path.dirname(os.path.abspath(inspect.stack()[3].filename))

    def create_exp_dir(self, exp_dir_name):
        directory = os.path.join(self.working_directory, exp_dir_name)
        proceed = True
        if os.path.exists(directory):
            is_empty, _, _ = self._check_dir_contents(directory)
            if not is_empty:
                message = (
                    f"The directory\n{directory}\n"
                    "already exists and is not empty.\n"
                    "Delete contents? [y/N]: "
                )
                delete = input(message).strip().lower()
                if delete == 'y':
                    shutil.rmtree(directory)
                    os.makedirs(directory)
                else:
                    proceed = False
        else:
            os.makedirs(directory)

        self.exp_dir = directory


        return proceed

    def create_root_chain_subdirs(self, n_chains):
        # Check if experiment directory exists before proceeding
        if not self.exp_dir:
            raise RuntimeError(
                "Experiment directory does not exist. Call 'create_exp_dir' first."
            )
        
        root_dirs = []
        for i in range(n_chains):
            root_dir = os.path.join(self.exp_dir, f"{self.root_chain_basename}_{i:02d}")
            os.makedirs(
                root_dir, exist_ok=True
            )
            root_dirs.append(root_dir)
        
        self.root_dirs = root_dirs

    def create_figures_dir(self, n_chains):

        directory = os.path.join(self.exp_dir, 'figures')
        os.makedirs(directory)
        root_dirs = []
        for i in range(n_chains):
            root_dir = os.path.join(directory, f"{self.root_chain_basename}_{i:02d}")
            os.makedirs(
                root_dir, exist_ok=True
            )
            root_dirs.append(root_dir)

        self.fig_dirs = root_dirs


        return 

    @staticmethod
    def _check_dir_contents(directory):
        subdir_count, file_count = 0, 0
        for _, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            subdir_count += len(dirs)
            file_count += len(files)
        return (subdir_count == 0 and file_count == 0), subdir_count, file_count
