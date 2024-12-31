import os
import json

import jax


class HMCRootConfig:
    _presets = {
        "default": {
            "n_chains": 3,
            "initial_state": "RANDOM",
            "seed_int_is": 1,
            "seed_int_chain": 1,
            "dtype": "SINGLE",
            "samples": [500, 200, 100],
            "temperature": [1, 1, 1],
            "lf_method": "HIGH_ORDER",
            "lf_step_min": 1e-4,
            "lf_step_max": 1e-2,
            "lf_steps": 6,
        },
    }

    def __init__(self, preset=None, **kwargs):
        config = self._presets.get(preset, {}).copy()
        config.update(kwargs)

        # Direct assignment with defaults
        self.n_chains = config.get("n_chains", 1)
        self.initial_state = config.get("initial_state", 1)
        self.seed_int_is = config.get("seed_int_is", 1)
        self.seed_int_chain = config.get("seed_int_chain", 1)
        self.dtype = config.get("dtype", "SINGLE")
        self.samples = config.get("samples", 100)
        self.temperature = config.get("temperature", 1)
        self.lf_method = config.get("lf_method", "HIGH_ORDER")
        self.lf_step_min = config.get("lf_step_min", 1e-4)
        self.lf_step_max = config.get("lf_step_max", 1e-2)
        self.lf_steps = config.get("lf_steps", 5)

        self.samples = [self.samples] if isinstance(self.samples, int) else self.samples

        self.n_epochs = len(self.samples)
        self.samples_per_chain = sum(self.samples)
        self.total_samples = self.samples_per_chain * self.n_chains

        self._validate_and_repeat()

    def _validate_and_repeat(self):
        for key, value in self.__dict__.items():
            if key not in [
                "temperature",
                "lf_steps",
                "lf_step_min",
                "lf_step_max",
                "lf_method",
            ]:
                continue
            # Validate length if value is a list
            if isinstance(value, list) and len(value) == 1:
                self.__dict__[key] = value * self.n_epochs
            # Validate length for other lists
            elif isinstance(value, list) and len(value) != self.n_epochs:
                raise ValueError(
                    f"Attribute '{key}' has length {len(value)}, but n_chains is {self.n_epochs}."
                )
            # Convert int/float to repeated lists
            else:
                self.__dict__[key] = self._repeat_to_list(value, self.n_epochs)
    @staticmethod
    def _repeat_to_list(value, N):
        list_val = [value] * N if isinstance(value, (int, float)) else value
        return list_val

    def save_to_json(self, directory):
        config_path = os.path.join(directory, "hmc_config.json")
        # Convert to dictionary
        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

        # Save as JSON
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)

    @classmethod
    def load_from_json(cls, filepath):
        """Load config from JSON file and return an HMCRootConfig instance."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(**data)
