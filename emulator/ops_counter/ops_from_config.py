import json
from copy import deepcopy
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from emulator.common import SupportedLayerTypes as Slt
from emulator.ops_counter.layer_configuration_containers import *

_TConfig = Dict[str, Dict[str, Any]]


def _get_input_bits_for_layer(config: Dict[str, Any], bottom: List[str]) -> List[int]:
    input_bits = []
    for bot in bottom:
        if 'activations_bits' in config[bot]:
            input_bits.append(config[bot]['activations_bits'])
        else:
            input_bits.append(max(_get_input_bits_for_layer(config, config[bot]['bottom'])))

    return input_bits


def _non_quant_layer_to_op_info_tuple(layer_cfg_data: Dict[str, Any]) -> Optional[BaseOperation]:
    layer_type = Slt(layer_cfg_data['type'])

    if layer_type in [Slt.LAYER_INPUT, Slt.LAYER_OUTPUT]:
        return None

    layer_input_shape = layer_cfg_data['input_shape']
    # layer_output_shape = layer_cfg_data['output_shape']

    if layer_type is Slt.LAYER_CONV2D:
        return Conv2D(
            input_size=layer_input_shape[1],
            kernel_shape=layer_cfg_data['weights_shape'],
            strides=layer_cfg_data['strides'][1:3],
            padding=layer_cfg_data['padding'].lower(),
            use_bias=True,  # Usually we always use bias
            activation=layer_cfg_data['activation'],
            weight_bits=32,
            activation_bits=32,
            input_bits=[32],
            output_bits=32,
        )

    if layer_type is Slt.LAYER_CONV2D_DEPTHWISE:
        return DepthWiseConv2D(
            input_size=layer_input_shape[1],
            kernel_shape=layer_cfg_data['weights_shape'],
            strides=layer_cfg_data['strides'][1:3],
            padding=layer_cfg_data['padding'].lower(),
            use_bias=True,  # Usually we always use bias
            activation=layer_cfg_data['activation'],
            weight_bits=32,
            activation_bits=32,
            input_bits=[32],
            output_bits=32,
        )

    if layer_type is Slt.LAYER_ADD:
        return Add(
            input_size=layer_input_shape[1],
            n_channels=layer_input_shape[-1],
            input_bits=[32],
            output_bits=32,
        )

    if layer_type is Slt.LAYER_REDUCE_MEAN:
        return GlobalAvg(
            input_size=layer_input_shape[1],
            n_channels=layer_input_shape[-1],
            input_bits=[32],
        )

    if layer_type is Slt.LAYER_FC:
        return FullyConnected(
            kernel_shape=layer_cfg_data['weights_shape'],
            use_bias=True,
            activation=layer_cfg_data['activation'],
            weight_bits=32,
            activation_bits=32,
            input_bits=[32],
            output_bits=32,
        )

    raise NotImplementedError(
        f'There is no counter associated with the layer type "{layer_type.value}"'
    )


def _layer_to_op_info_tuple(layer_cfg_data: Dict[str, Any], config: Dict[str, Any]) -> Optional[BaseOperation]:
    layer_type = Slt(layer_cfg_data['type'])

    if layer_type in [Slt.LAYER_INPUT, Slt.LAYER_OUTPUT]:
        return None

    layer_input_shape = layer_cfg_data['input_shape']

    if layer_type is Slt.LAYER_CONV2D:
        return Conv2D(
            input_size=layer_input_shape[1],
            kernel_shape=layer_cfg_data['weights_shape'],
            strides=layer_cfg_data['strides'][1:3],
            padding=layer_cfg_data['padding'].lower(),
            use_bias=True,  # Usually we always use bias
            activation=layer_cfg_data['activation'],
            weight_bits=layer_cfg_data['weights_bits'],
            activation_bits=layer_cfg_data['activations_bits'],
            input_bits=_get_input_bits_for_layer(config, layer_cfg_data['bottom']),
            output_bits=layer_cfg_data['activations_bits'],
        )

    if layer_type is Slt.LAYER_CONV2D_DEPTHWISE:
        return DepthWiseConv2D(
            input_size=layer_input_shape[1],
            kernel_shape=layer_cfg_data['weights_shape'],
            strides=layer_cfg_data['strides'][1:3],
            padding=layer_cfg_data['padding'].lower(),
            use_bias=True,  # Usually we always use bias
            activation=layer_cfg_data['activation'],
            weight_bits=layer_cfg_data['weights_bits'],
            activation_bits=layer_cfg_data['activations_bits'],
            input_bits=_get_input_bits_for_layer(config, layer_cfg_data['bottom']),
            output_bits=layer_cfg_data['activations_bits'],
        )

    if layer_type is Slt.LAYER_ADD:
        return Add(
            input_size=layer_input_shape[1],
            n_channels=layer_input_shape[-1],
            input_bits=_get_input_bits_for_layer(config, layer_cfg_data['bottom']),
            output_bits=layer_cfg_data['activations_bits'],
        )

    if layer_type is Slt.LAYER_REDUCE_MEAN:
        return GlobalAvg(
            input_size=layer_input_shape[1],
            n_channels=layer_input_shape[-1],
            input_bits=_get_input_bits_for_layer(config, layer_cfg_data['bottom']),
        )

    if layer_type is Slt.LAYER_FC:
        return FullyConnected(
            kernel_shape=layer_cfg_data['weights_shape'],
            use_bias=True,
            activation=layer_cfg_data['activation'],
            weight_bits=layer_cfg_data['weights_bits'],
            activation_bits=layer_cfg_data['activations_bits'],
            input_bits=_get_input_bits_for_layer(config, layer_cfg_data['bottom']),
            output_bits=layer_cfg_data['activations_bits'],
        )

    raise NotImplementedError(
        f'There is no counter associated with the layer type "{layer_type.value}"'
    )


def get_countable_ops(
        configuration: Union[str, Path, _TConfig],
        is_quantized: bool = False
) -> List[Tuple[str, BaseOperation]]:
    """Get list of BaseOperation suitable with ops counter"""
    if isinstance(configuration, dict):
        configuration = deepcopy(configuration)
    else:
        configuration_path = Path(configuration)
        with configuration_path.open('r') as file:
            configuration = json.load(file)

    ops = []

    for layer_name, layer_cfg in configuration.items():
        if is_quantized:
            layer_info_tuple = _layer_to_op_info_tuple(layer_cfg, configuration)
        else:
            layer_info_tuple = _non_quant_layer_to_op_info_tuple(layer_cfg)
        if layer_info_tuple is not None:
            ops.append((layer_name, layer_info_tuple))

    return ops
