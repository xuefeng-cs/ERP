from collections.abc import Mapping
import datetime
import logging
import os

import hydra

logger = logging.getLogger(__name__)


def get_output_dir() -> str:
    try:
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        return hydra_cfg["runtime"]["output_dir"]
    except ValueError:
        return os.path.join(
            "outputs",
            datetime.date.today().strftime("%Y-%m-%d"),
            datetime.datetime.now().time().strftime("%H-%M-%S"),
        )


# def update_config(config: Mapping, new_config: Mapping, force=False) -> Mapping:
#     """Update config with new_config, in place, recursively"""
#     for key, value in new_config.items():
#         if isinstance(value, Mapping):
#             update_config(config[key], value)
#         else:
#             if value is None:
#                 continue
#             if key in config and config[key] is not None and not force:
#                 raise ValueError(
#                     f"Key {key} already has value {config[key]} and force is False"
#                 )
#             config[key] = value
#             logger.info(f"Update {key} to {value}")
#     return config
