from .tbsm import build as build_tbsm
from .common import module_state_dict


BUILDERS = {
    "tbsm": build_tbsm,
}

__all__ = ["build_training_model", "module_state_dict"]


def build_training_model(cfg):
    method_name = cfg["method"]["name"]
    if method_name not in BUILDERS:
        raise ValueError(f"Unknown method builder: {method_name}")
    return BUILDERS[method_name](cfg)
