import pickle
from pathlib import Path
from typing import Dict
from typing import Tuple
from typing import Union

import numpy as np

import emulator
from emulator.common.data_utils import load_weights
from emulator.fakequant.calibrators import ThresholdsMappedData
from emulator.fakequant.quantize_utils import nudge_parameters_np_ex
from emulator.integer.integer_layers import ScalarQuantizationParameters
from emulator.integer.integer_layers import VectorQauntizationParameters

_TPath = Union[str, Path]


def _load_thresholds(path: Union[str, Path]) -> Tuple[ThresholdsMappedData, ThresholdsMappedData]:
    path = Path(path)
    with path.open('rb') as file:
        aths , wths = pickle.load(file)

    if not isinstance(aths, ThresholdsMappedData):
        aths = ThresholdsMappedData(aths)

    if not isinstance(wths, ThresholdsMappedData):
        wths = ThresholdsMappedData(wths)

    return aths, wths


def _get_vector_quantization_data(
        min_max_th: Tuple[np.ndarray, np.ndarray],
        bits: int,
        narrow_range: bool,
) -> VectorQauntizationParameters:
    n_min, n_max, n_scale, n_zero = nudge_parameters_np_ex(min_max_th[0], min_max_th[1], bits, narrow_range)
    return VectorQauntizationParameters(n_min, n_max, n_scale.squeeze(), n_zero.squeeze())


def _get_scalar_quantization_data(
        min_max_th: Tuple[np.ndarray, np.ndarray],
        bits: int,
        narrow_range: bool,
) -> ScalarQuantizationParameters:
    n_min, n_max, n_scale, n_zero = nudge_parameters_np_ex(min_max_th[0], min_max_th[1], bits,narrow_range)
    return ScalarQuantizationParameters(n_min.item(), n_max.item(), n_scale.item(), int(n_zero.item()))


def get_cfg_weights_and_quant_data(
        reg_cfg_path: _TPath,
        quant_cfg_path: _TPath,
        weights_path: _TPath,
        thresholds_path: _TPath,
        img_size: int,
):
    a_ths, w_ths = _load_thresholds(thresholds_path)

    a_ths_dict = a_ths.thresholds_dict
    w_ths_dict = w_ths.thresholds_dict

    reg_cfg = emulator.common.cfg_utils.load_cfg(reg_cfg_path)
    quant_cfg = emulator.common.cfg_utils.load_cfg(quant_cfg_path)

    reg_cfg['input_node_1']['shape'] = [None, img_size, img_size, 3]
    quant_cfg['input_node_1']['shape'] = [None, img_size, img_size, 3]

    weights = load_weights(weights_path)

    quantization_data: Dict[str, Dict[str, Union[ScalarQuantizationParameters, VectorQauntizationParameters]]] = {}

    cfg = quant_cfg

    def _get_quant_params_of_previous_layer(current_layer_name: str) -> ScalarQuantizationParameters:
        ln = current_layer_name
        l_cfg = cfg[ln]

        while True:
            ln = l_cfg['bottom'][0]
            l_cfg = cfg[ln]
            if 'activations_bits' in l_cfg:
                break

        return _get_scalar_quantization_data(
            a_ths_dict[ln],
            l_cfg['activations_bits'],
            False,
        )

    for ln, lcfg in cfg.items():
        if lcfg['type'] == 'input_node':
            quantization_data[ln] = {
                'output_quant_data': _get_scalar_quantization_data(
                    a_ths_dict[ln],
                    cfg[ln]['activations_bits'],
                    False,
                ),
            }

        elif lcfg['type'] in ['conv2d_layer', 'conv2d_depthwise_layer', 'fully_connected_layer']:
            quantization_data[ln] = {
                'input_quant_data': _get_quant_params_of_previous_layer(ln),
                'output_quant_data': _get_scalar_quantization_data(
                    a_ths_dict[ln],
                    cfg[ln]['activations_bits'],
                    False,
                ),
                'weights_quant_data': _get_vector_quantization_data(
                    w_ths_dict[ln],
                    cfg[ln]['weights_bits'],
                    cfg[ln]['weights_narrow_range'],
                ),
            }
        elif lcfg['type'] in ['mul_layer', 'add_layer']:
            inputs = lcfg['bottom']
            quantization_data[ln] = {
                'input_1_quant_data': _get_scalar_quantization_data(
                    a_ths_dict[inputs[0]],
                    cfg[inputs[0]]['activations_bits'],
                    False,
                ),
                'input_2_quant_data': _get_scalar_quantization_data(
                    a_ths_dict[inputs[1]],
                    cfg[inputs[1]]['activations_bits'],
                    False,
                ),
                'output_quant_data': _get_scalar_quantization_data(
                    a_ths_dict[ln],
                    cfg[ln]['activations_bits'],
                    False,
                ),
            }

    return reg_cfg, weights, quantization_data
