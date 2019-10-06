__all__ = [
    'TFLiteGpuInterpreter',
]

from copy import deepcopy
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Union

from emulator.common.cfg_utils import load_cfg
from emulator.common.data_types import TWeightsCfg
from emulator.integer.integer_meta_layer import IntegerMetaLayer
from emulator.layers import BaseGraphLayer
from emulator.regular import RegularModel

_DEFAULT_QUANTIZED_STATS = (128., 127.5)


class TFLiteGpuInterpreter(RegularModel):

    def __init__(
            self,
            cfg_or_cfg_path: Union[str, Path, dict],
            weights: TWeightsCfg,
            quantization_data: Dict[str, Dict[str, Any]],
            new_input_shape=None,
    ):
        cfg = load_cfg(cfg_or_cfg_path)

        if new_input_shape is not None:
            cfg = _get_cfg_with_new_input_shape(cfg, new_input_shape)

        weights = deepcopy(weights)
        quantization_data = deepcopy(quantization_data)

        for ln in quantization_data:
            weights.setdefault(ln, {}).update(quantization_data[ln])

        super().__init__(
            cfg_or_cfg_path=cfg,
            weights=weights,
        )

    @staticmethod
    def _create_layer(layer_inputs: List[BaseGraphLayer], layer_cfg: dict) -> BaseGraphLayer:
        return IntegerMetaLayer.create_layer(inputs=layer_inputs, **layer_cfg)


def _get_cfg_with_new_input_shape(cfg, new_input_shape):
    cfg = load_cfg(cfg)
    for ln, ld in cfg.items():
        if ln == 'input_node_1':
            ld['shape'] = list(new_input_shape)
            break
    return cfg
