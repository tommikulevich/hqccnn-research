"""Grid search helpers."""
import copy
from itertools import product
from typing import Callable

from config.schema import Config


def grid_search(cfg: Config, run_fn: Callable):
    """Perform grid search over hyperparameters."""
    grid_params = cfg.search.params
    if not grid_params:
        raise ValueError("No parameters provided for grid search.")

    keys = list(grid_params.keys())
    values_list = [
        v if isinstance(v, (list, tuple)) else [v]
        for v in grid_params.values()
    ]

    print("Possibilities: ", zip(keys, product(*values_list)))
    for combo in product(*values_list):
        cfg_copy = copy.deepcopy(cfg)
        for key, val in zip(keys, combo):
            parts = key.split('.')
            if len(parts) == 3 and parts[1] == 'params':
                section, _, param_name = parts
                getattr(cfg_copy, section).params[param_name] = val
            else:
                raise ValueError(f"Unsupported search parameter key: {key}")

        run_fn(cfg_copy)
