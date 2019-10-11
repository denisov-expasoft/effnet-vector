__all__ = ['rescale_model_weights']

from copy import deepcopy
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Union

import numpy as np

from emulator.common.cfg_utils import load_cfg
from emulator.common.graph_walker import GraphWalker
from emulator.common.layers_conf_parameters import LayerConfigurationParameters as Lcp
from emulator.common.layers_types import SupportedLayerTypes as Slt

TConfig = Dict[str, Dict[str, Any]]
TWeights = Dict[str, Dict[str, np.ndarray]]

_DWS_REDUCTION_AXIS = (0, 1, 3)
_LINEAR_ACTIVATIONS = (None, 'relu')

# Parameter controls how close to the upper activation limit
# the layer output can be scaled
_ACTIVATIONS_SAFE_CLAMP_DISTANCE = 0.05


class DwsConvPairInfo(NamedTuple):
    dws_name: str
    conv_name: str
    activation_uper_limit: Optional[np.array]


_TLayerPairs = List[DwsConvPairInfo]


class DwsRescaleRule:
    """Object that containt rescale information for individual Dws->Conv pairs"""

    def __init__(
            self,
            dws_layer_name: str,
            conv_layer_name: str,
            scale: np.ndarray,
    ):
        self._dws_layer_name = dws_layer_name
        self._conv_layer_name = conv_layer_name
        self._scale = np.array(scale)

    @property
    def dws_scale(self) -> np.ndarray:
        return np.squeeze(self._scale).reshape((-1, 1))

    @property
    def bias_scale(self) -> np.ndarray:
        return np.squeeze(self._scale).reshape(-1)

    def apply_scale(
            self,
            model_weights: TWeights,
    ) -> None:
        dws_weights_block = model_weights[self._dws_layer_name]
        conv_weights_block = model_weights[self._conv_layer_name]

        dws_weights_block['weights'] = dws_weights_block['weights'] * self.dws_scale
        dws_weights_block['bias'] = dws_weights_block['bias'] * self.bias_scale
        conv_weights_block['weights'] = conv_weights_block['weights'] * np.reciprocal(self.dws_scale)


def _get_layer_type(name: str, cfg: TConfig) -> Slt:
    layer_type = cfg[name][Lcp.ATTR_COMMON_TYPE.value]
    layer_type = Slt(layer_type)
    return layer_type


def _select_pairs_to_rescale(configuration: TConfig) -> _TLayerPairs:
    linear_pairs = []

    graph_walker = GraphWalker(configuration)
    forward_map = graph_walker.copy_forward_mapping()

    for layer_name, layer_data in configuration.items():
        if _get_layer_type(layer_name, configuration) != Slt.LAYER_CONV2D_DEPTHWISE:
            continue

        next_layer_name = forward_map[layer_name]
        if len(next_layer_name) != 1:
            continue

        next_layer_name = list(next_layer_name)[0]

        if _get_layer_type(next_layer_name, configuration) != Slt.LAYER_CONV2D:
            continue

        dws_activation = layer_data.get(Lcp.ATTR_NN_ACTIVATION_TYPE.value)

        if dws_activation in _LINEAR_ACTIVATIONS:
            linear_pairs.append(DwsConvPairInfo(layer_name, next_layer_name, None))

    return linear_pairs


def _get_scaling_rules_for_dws_relu_conv(
        model_weights: TWeights,
        list_of_layers_pairs: _TLayerPairs,
) -> List[DwsRescaleRule]:
    rescale_data = []

    for dws_name, conv_name, _ in list_of_layers_pairs:
        dws_kernel = model_weights[dws_name]['weights']
        filter_maxabs_vals = np.max(np.abs(dws_kernel), _DWS_REDUCTION_AXIS)
        scale = np.max(filter_maxabs_vals) / filter_maxabs_vals

        rescale_data.append(
            DwsRescaleRule(dws_name, conv_name, scale)
        )

    return rescale_data


def rescale_model_weights(
        weights: TWeights,
        cfg_or_cfg_path: Union[str, Path, dict],
) -> TWeights:
    cfg = load_cfg(cfg_or_cfg_path)
    processed_weights = deepcopy(weights)

    linear_pairs = _select_pairs_to_rescale(cfg)

    # Process the DWS -> [ReLU] -> Conv case
    if linear_pairs:
        scales_rules = _get_scaling_rules_for_dws_relu_conv(
            weights,
            linear_pairs,
        )
        for scales_rule in scales_rules:
            scales_rule.apply_scale(processed_weights)

    return processed_weights
