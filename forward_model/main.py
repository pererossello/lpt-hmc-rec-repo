import inspect

from .bias import get_forward_bias
from .lpt import get_forward_lpt

def get_forward_model(arg_dict):

    # Extract arguments and their defaults from the two functions
    lpt_signature = inspect.signature(get_forward_lpt)
    bias_signature = inspect.signature(get_forward_bias)

    lpt_args = {
        key: param.default if param.default is not param.empty else None
        for key, param in lpt_signature.parameters.items()
    }
    bias_args = {
        key: param.default if param.default is not param.empty else None
        for key, param in bias_signature.parameters.items()
    }

    # Merge `arg_dict` with defaults, prioritizing `arg_dict` values
    lpt_args = {key: arg_dict.get(key, default) for key, default in lpt_args.items()}
    bias_args = {key: arg_dict.get(key, default) for key, default in bias_args.items()}

    # Call the respective functions
    forward_lpt = get_forward_lpt(**lpt_args)
    forward_bias = get_forward_bias(**bias_args)

    # Construct the forward model
    forward_model = lambda x: forward_bias(forward_lpt(x))

    return forward_model
