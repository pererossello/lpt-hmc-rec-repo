import os
import json
import jax

class HMCConfig:
    _presets = {
        "default": {
            "seed_int_sample": 1,
            "samples": 500,
            "forward_model": "FROM_DATA", 
            "temperature": 1,
            "lf_method": "HIGH_ORDER",
            "lf_step_min": 1e-4,
            "lf_step_max": 1e-2,
            "lf_steps": 6,
            "dtype": "SINGLE",
        },
    }

    """
    'forward_model' and 'lf_method' are the same for all epochs 
    """
    def __init__(self, preset='default', **kwargs):
        config = self._presets.get(preset, {}).copy()
        config.update(kwargs)

        self._normalize_and_assign(config)

        self.n_epochs = len(self.samples)
        self.total_samples = sum(self.samples)

    def to_dict(self):
        return vars(self).copy()

    def _normalize_and_assign(self, config):
        samples = config.get("samples")
        if isinstance(samples, int):
            samples = [samples]

        n_samples = len(samples)
        self.samples = samples

        # Parameters to synchronize with samples length
        sync_keys = ["temperature", "lf_step_min", "lf_step_max", "lf_steps"]

        for key in sync_keys:
            value = config.get(key)

            # If int or float, repeat to match length of samples
            if isinstance(value, (int, float)):
                value = [value] * n_samples

            # If list, check if the length matches samples
            elif isinstance(value, list):
                if len(value) == 1:
                    value = value * n_samples  # Expand if list of length 1
                elif len(value) != n_samples:
                    raise ValueError(
                        f"Length of '{key}' ({len(value)}) does not match "
                        f"length of 'samples' ({n_samples})."
                    )
            else:
                raise TypeError(f"Invalid type for '{key}': {type(value)}")

            # Set the normalized value
            setattr(self, key, value)

        self.forward_model = config.get("forward_model")
        self.lf_method = config.get("lf_method")
        self.dtype = config.get('dtype')
        self.seed_int_sample = config.get("seed_int_sample")


    # def _validate_params(self, config):
    #     allowed_keys = self._presets["default"].keys()
    #     for key in config:
    #         if key not in allowed_keys:
    #             raise ValueError(f"Invalid configuration key: {key}")
            


    def __repr__(self):
        return f"HMCConfig({vars(self)})"