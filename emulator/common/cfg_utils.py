__all__ = [
    'load_cfg',
    'configuration_has_quantized_layers',
    'layer_has_quantization_options',
    'ACTIVATIONS_QUANT_PARAMS',
    'WEIGHTS_QUANT_PARAMS',
    'ALL_QUANT_PARAMS',
]

import json
from copy import deepcopy
from pathlib import Path
from typing import Any
from typing import Dict
from typing import FrozenSet
from typing import Union

from emulator.common.layers_conf_parameters import LayerConfigurationParameters as Lcp


ACTIVATIONS_QUANT_PARAMS = frozenset({
    Lcp.QUANT_ACTIVATIONS_BITS.value,
})

WEIGHTS_QUANT_PARAMS = frozenset({
    Lcp.QUANT_WEIGHTS_BITS.value,
    Lcp.QUANT_WEIGHTS_THRESHOLDS.value,
    Lcp.QUANT_WEIGHTS_NARROW_RANGE.value,
})

ALL_QUANT_PARAMS: FrozenSet[str] = ACTIVATIONS_QUANT_PARAMS | WEIGHTS_QUANT_PARAMS


def load_cfg(cfg_or_cfg_path: Union[str, Path, dict]) -> dict:
    if isinstance(cfg_or_cfg_path, dict):
        return deepcopy(cfg_or_cfg_path)

    if not isinstance(cfg_or_cfg_path, (str, Path)):
        raise TypeError(
            'Configuration data must be provided via a dictionary or a path '
            'to the file containing the Network configuration data'
        )

    cfg_path = Path(cfg_or_cfg_path)

    if not cfg_path.exists():
        raise FileNotFoundError(
            f'The specified configuration file "{cfg_path}" '
            f'does not exist'
        )

    with cfg_path.open('r') as cfg_file:
        return json.load(cfg_file)


def configuration_has_quantized_layers(cfg_or_cfg_path: Union[str, Path, dict]) -> bool:
    network_cfg: Dict[str, Dict[str, Any]] = load_cfg(cfg_or_cfg_path)

    for node_cfg_data in network_cfg.values():
        if layer_has_quantization_options(node_cfg_data):
            return True
    return False


def layer_has_quantization_options(node_cfg: Dict[str, Any]) -> bool:
    return not ALL_QUANT_PARAMS.isdisjoint(node_cfg)
