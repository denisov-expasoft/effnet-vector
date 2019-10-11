__all__ = [
    'FQMetaLayerEnvelope',
    'FQRequantizationMetaLayer',
    'FQMetaLayerWithActivation',
    'FQMatrixOpsMetaLayer',
    'FAKEQUANT_METALAYERS_REGISTRY',
]

from copy import deepcopy
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
import tensorflow as tf

from emulator.common import LayerConfigurationParameters as Lcp
from emulator.common import Registry
from emulator.common import SupportedLayerTypes as Slt
from emulator.common import any_tensor
from emulator.common import drop_nones
from emulator.common import exceptions
from emulator.common.data_types import TArrayOrTensor
from emulator.common.data_types import TMinMaxThresholdsData
from emulator.fakequant.calibrators import BaseCalibrator
from emulator.fakequant.calibrators import MinMaxCalibrator
from emulator.fakequant.calibrators import WEIGHTS_CALIBRATORS_REGISTRY
from emulator.fakequant.fakequant_layers import FakeQuantLayer
from emulator.fakequant.quantize_utils import discretize_array_or_tensor_v2
from emulator.fakequant.quantize_utils import nudge_parameters_np
from emulator.layers import BaseGraphLayer
from emulator.layers import BaseMetaLayer
from emulator.layers import LAYERS_REGISTRY
from emulator.layers import ReluActivationLayer
from emulator.layers import SwishActivationLayer
from emulator.layers import SigmoidActivationLayer

_SUPPORTED_ACTIVATIONS = (None, 'relu', 'swish', 'sigmoid')

FAKEQUANT_METALAYERS_REGISTRY = Registry(key_type=Slt)


class FQMetaLayerEnvelope(BaseMetaLayer):

    def __init__(self, **layer_cfg):
        self._input_scale = None
        self._output_scale = None

        layer_cfg.pop(Lcp.ATTR_INPUT_SHAPE.value, None)
        layer_cfg.pop(Lcp.ATTR_OUTPUT_SHAPE.value, None)

        super().__init__(layers_registry=LAYERS_REGISTRY, **layer_cfg)

    @property
    def output_scale(self) -> Optional[TArrayOrTensor]:
        return self._output_scale

    @property
    def input_quantized(self):
        return self._input_scale is not None

    def _maybe_get_output_scale_from_prev_layers(self) -> Optional[TArrayOrTensor]:

        prev_scales = [
            getattr(input_layer, 'output_scale', None)
            for input_layer in self.inputs
        ]

        def is_scale_scalar(scale: TArrayOrTensor) -> bool:
            if isinstance(scale, tf.Tensor):
                scale_size = scale.shape.as_list()
                scale_size = np.prod(scale_size)
            else:
                scale_size = scale.size

            return scale_size == 1

        def maybe_squeeze_scale(scale: TArrayOrTensor) -> TArrayOrTensor:
            """Scales transferred from previous layers can have
            incompatible number of dimensions. We squeeze it to
            prevent unwanted behaviour of numpy / tensorflow broadcasting
            """
            if is_scale_scalar(scale):
                if isinstance(scale, tf.Tensor):
                    if scale.shape.as_list():
                        return tf.squeeze(scale, name='input_scale')
                else:
                    # We do that only if the scale is not a tf.Tensor
                    return np.squeeze(scale)

            if isinstance(scale, tf.Tensor):
                return tf.identity(scale, name='input_scale')

            return scale

        # If the current layer have a single input,
        # we can simply forward its scale
        if len(prev_scales) == 1:
            if prev_scales[0] is None:
                return None
            return maybe_squeeze_scale(prev_scales[0])

        # If we have two or more inputs we must check that they are all
        # whether quantized or not
        undefined_scales = [prev_scale is None for prev_scale in prev_scales]

        # If all input scales are undefined, then we deal with non-quantized model
        if all(undefined_scales):
            return None

        # If some inputs are quantized but not all,
        # we deal with the inconsistent model configuration
        if any(undefined_scales):
            raise exceptions.GraphConfigError(
                f'Multi-input layer receives quantized inputs mixed with non-quantized ones'
            )

        # Layers with several inputs (like Add or Concat) accept data
        # quantized with the scalar scale
        all_scale_scalar = all(
            is_scale_scalar(prev_scale)
            for prev_scale in prev_scales
        )

        if not all_scale_scalar:
            raise exceptions.GraphConfigError(
                f'Multi-input layer receives inputs with non-scalar quantization'
            )

        # We suppose that all input scales are correct (equal)
        # (we do not check actual scales here)
        return maybe_squeeze_scale(prev_scales[0])

    def _create_backend_operations(self) -> tf.Tensor:
        self._input_scale = self._maybe_get_output_scale_from_prev_layers()
        self._prepare_layer_data()

        return super()._create_backend_operations()

    def _prepare_layer_data(self) -> None:
        self._output_scale = self._input_scale

    def get_meaningful_properties(self) -> Dict[str, Any]:
        result = {
            'input_scale': self._copy_array_or_tensor_data(self._input_scale),
            'output_scale': self._copy_array_or_tensor_data(self._output_scale),
        }

        return drop_nones(result)


@FAKEQUANT_METALAYERS_REGISTRY.add_item_decorator([
    Slt.LAYER_INPUT,
    Slt.LAYER_ADD,
    Slt.LAYER_MUL,
])
class FQRequantizationMetaLayer(FQMetaLayerEnvelope):

    def __init__(self, activations_thresholds_data: TMinMaxThresholdsData = None, **layer_cfg):

        self._activations_bits = layer_cfg.pop(Lcp.QUANT_ACTIVATIONS_BITS.value, None)
        self._activations_thresholds: TMinMaxThresholdsData = activations_thresholds_data
        self._outputs_quantized = self._activations_bits is not None

        self._activations_nudged_min = None
        self._activations_nudged_max = None

        if self._outputs_quantized and self._activations_thresholds is None:
            raise ValueError(
                'Quantization thresholds for activations are not provided '
                'for a layer with the specified quantization parameters'
            )

        super().__init__(**layer_cfg)

    def _maybe_calibrate_output(self) -> None:
        if not self._outputs_quantized:
            return

        calibrator = MinMaxCalibrator()

        min_thresholds, max_thresholds = self._activations_thresholds
        nudged_min, nudged_max, scale = nudge_parameters_np(
            min_thresholds=min_thresholds,
            max_thresholds=max_thresholds,
            bits=self._activations_bits,
            narrow_range=False,
        )

        self._activations_nudged_min = nudged_min
        self._activations_nudged_max = nudged_max
        self._activations_scale = calibrator.to_output_format_np(scale)

        self._output_scale = self._activations_scale

    def _quantize_outputs(self, output_layer: BaseGraphLayer) -> BaseGraphLayer:

        if not self._outputs_quantized:
            return output_layer

        fake_quantization_layer_instance = FakeQuantLayer(
            scale=self._activations_scale,
            nudged_min=self._activations_nudged_min,
            nudged_max=self._activations_nudged_max,
        )

        with tf.name_scope(f'discrete_output'):
            quantized_output = fake_quantization_layer_instance(output_layer)

        return quantized_output

    def _prepare_layer_data(self):
        self._output_scale = self._input_scale
        self._maybe_calibrate_output()

    def _process_output_layer(self, output_layer: BaseGraphLayer) -> BaseGraphLayer:
        if not self._outputs_quantized:
            return output_layer

        with tf.name_scope('quantize_output'):
            return self._quantize_outputs(output_layer)

    @property
    def activations_thresholds(self) -> Optional[Tuple[TArrayOrTensor, TArrayOrTensor]]:
        return self._activations_thresholds

    def get_meaningful_properties(self) -> Dict[str, Any]:
        meaningful_properties = super().get_meaningful_properties()

        if self.activations_thresholds is not None:
            a_min_ths, a_max_ths = self.activations_thresholds
            a_min_ths = self._copy_array_or_tensor_data(a_min_ths)
            a_max_ths = self._copy_array_or_tensor_data(a_max_ths)
            activations_thresholds = a_min_ths, a_max_ths

            meaningful_properties['activations_thresholds'] = activations_thresholds

        return meaningful_properties


class FQMetaLayerWithActivation(FQRequantizationMetaLayer):

    def __init__(self, **layer_cfg):

        self._activation_function = layer_cfg.pop(Lcp.ATTR_NN_ACTIVATION_TYPE.value)

        if self._activation_function not in _SUPPORTED_ACTIVATIONS:
            raise TypeError(
                f'Unsupported value for the activation function: {self._activation_function}'
            )

        super().__init__(**layer_cfg)

    def _process_output_layer(self, output_layer: BaseGraphLayer) -> BaseGraphLayer:
        if self._activation_function is not None:
            if self._activation_function == 'relu':
                activation_layer_instance = ReluActivationLayer()
                output_layer = activation_layer_instance(output_layer)
            elif self._activation_function == 'swish':
                activation_layer_instance = SwishActivationLayer()
                output_layer = activation_layer_instance(output_layer)
            elif self._activation_function == 'sigmoid':
                activation_layer_instance = SigmoidActivationLayer()
                output_layer = activation_layer_instance(output_layer)
            else:
                raise NotImplementedError(f'Behaviour of "{self._activation_function}" is not implemented')

        if self._outputs_quantized:
            with tf.name_scope('quantize_output'):
                output_layer = self._quantize_outputs(output_layer)

        return output_layer


@FAKEQUANT_METALAYERS_REGISTRY.add_item_decorator([
    Slt.LAYER_CONV2D,
    Slt.LAYER_CONV2D_DEPTHWISE,
    Slt.LAYER_FC,
])
class FQMatrixOpsMetaLayer(FQMetaLayerWithActivation):

    def __init__(self, weights_thresholds_data: TMinMaxThresholdsData = None, **layer_cfg):

        self._weights_calibration_type = layer_cfg.pop(Lcp.QUANT_WEIGHTS_THRESHOLDS.value, None)
        self._weights_bits = layer_cfg.pop(Lcp.QUANT_WEIGHTS_BITS.value, None)
        self._weights_narrow_range = layer_cfg.pop(Lcp.QUANT_WEIGHTS_NARROW_RANGE.value, None)
        self._weights_thresholds: TMinMaxThresholdsData = weights_thresholds_data

        self._weights_quantized = (
                self._weights_calibration_type is not None or
                self._weights_bits is not None or
                self._weights_narrow_range is not None
        )

        if self._weights_quantized:
            if self._weights_calibration_type is None:
                raise exceptions.GraphLayerCfgError(
                    f'The specified list of weights quantization parameters is not complete. '
                    f'"{Lcp.QUANT_WEIGHTS_THRESHOLDS.value}" is missing.'
                )

            if self._weights_bits is None:
                raise exceptions.GraphLayerCfgError(
                    f'The specified list of weights quantization parameters is not complete. '
                    f'"{Lcp.QUANT_WEIGHTS_BITS.value}" is missing.'
                )

            if self._weights_narrow_range is None:
                raise exceptions.GraphLayerCfgError(
                    f'The specified list of weights quantization parameters is not complete. '
                    f'"{Lcp.QUANT_WEIGHTS_NARROW_RANGE.value}" is missing.'
                )

            if self._weights_thresholds is None:
                raise ValueError(
                    f'Thresholds for wheights of the quantized layer are missing.'
                )

            if Lcp.QUANT_ACTIVATIONS_BITS.value not in layer_cfg:
                raise exceptions.GraphLayerCfgError(
                    f'Activations quantization parameters are not specified for '
                    f'the layer with quantized weights'
                )

        self._weights = layer_cfg.pop('weights')
        self._bias = layer_cfg.pop('bias', None)
        self._discrete_weights, self._discrete_bias = None, None

        self._weights_scale = None
        self._weights_output_scale = None
        self._bias_scale = None

        self._weights_nudged_min = None
        self._weights_nudged_max = None

        super().__init__(**layer_cfg)

    def _quantize_weights(self) -> None:
        if not any_tensor(
                self._weights,
                self._weights_scale,
                self._weights_nudged_min,
                self._weights_nudged_max,
        ):
            _, discrete_weights = discretize_array_or_tensor_v2(
                data=self._weights,
                scale=self._weights_scale,
                nudged_min=self._weights_nudged_min,
                nudged_max=self._weights_nudged_max,
            )
            self._discrete_weights = self.maybe_save_const(
                discrete_weights,
                name='discrete_weights',
                dtype=tf.float32,
            )

        else:
            with tf.name_scope('discrete_weights'):
                _, self._discrete_weights = discretize_array_or_tensor_v2(
                    data=self.maybe_save_const(self._weights, name='weights', dtype=tf.float32),
                    scale=self.maybe_save_const(self._weights_scale, name='scale', dtype=tf.float32),
                    nudged_min=self.maybe_save_const(self._weights_nudged_min, name='nudged_min', dtype=tf.float32),
                    nudged_max=self.maybe_save_const(self._weights_nudged_max, name='nudged_max', dtype=tf.float32),
                )

    def _quantize_bias(self) -> None:
        if not any_tensor(
                self._bias,
                self._bias_scale,
        ):
            _, discrete_bias = discretize_array_or_tensor_v2(
                data=self._bias,
                scale=self._bias_scale,
                nudged_min=None,
                nudged_max=None,
            )
            self._discrete_bias = self.maybe_save_const(
                discrete_bias,
                name='discrete_bias',
                dtype=tf.float32,
            )

        else:
            with tf.name_scope('discrete_bias'):
                _, self._discrete_bias = discretize_array_or_tensor_v2(
                    data=self.maybe_save_const(self._bias, name='bias', dtype=tf.float32),
                    scale=self.maybe_save_const(self._bias_scale, name='scale', dtype=tf.float32),
                    nudged_min=None,
                    nudged_max=None,
                )

    def _maybe_quantize_weights_and_bias(self):
        if not self._weights_quantized:
            return

        self._quantize_weights()

        if self._bias is not None:
            self._quantize_bias()

    def _maybe_calibrate_weights(self) -> None:

        if not self._weights_quantized:
            return

        calibrator: BaseCalibrator = WEIGHTS_CALIBRATORS_REGISTRY[self._weights_calibration_type]()

        min_thresholds, max_thresholds = self._weights_thresholds
        nudged_min, nudged_max, scale = nudge_parameters_np(
            min_thresholds=min_thresholds,
            max_thresholds=max_thresholds,
            bits=self._weights_bits,
            narrow_range=self._weights_narrow_range,
        )

        self._weights_nudged_min = nudged_min
        self._weights_nudged_max = nudged_max

        self._weights_scale = scale
        self._weights_output_scale = calibrator.to_output_format_np(self._weights_scale)

        self._bias_scale = self._create_bias_scale()

        self._output_scale = self._bias_scale

    def _create_bias_scale(self) -> TArrayOrTensor:
        return np.asarray(self._input_scale * self._weights_output_scale)

    def _prepare_cfg(self) -> dict:
        layer_cfg = deepcopy(self._base_layer_cfg)

        if self._discrete_weights is not None:
            layer_cfg['weights'] = self._discrete_weights
        elif self._weights is not None:
            layer_cfg['weights'] = self._weights

        if self._discrete_bias is not None:
            layer_cfg['bias'] = self._discrete_bias
        elif self._bias is not None:
            layer_cfg['bias'] = self._bias

        return layer_cfg

    def _prepare_layer_data(self):
        self._output_scale = self._input_scale
        self._maybe_calibrate_weights()
        self._maybe_calibrate_output()
        self._maybe_quantize_weights_and_bias()

    @property
    def weights_thresholds(self) -> Optional[Tuple[TArrayOrTensor, TArrayOrTensor]]:
        return self._weights_thresholds

    def get_meaningful_properties(self) -> Dict[str, Any]:
        meaningful_properties = super().get_meaningful_properties()

        if self.weights_thresholds is not None:
            w_min_ths, w_max_ths = self.weights_thresholds
            w_min_ths = self._copy_array_or_tensor_data(w_min_ths)
            w_max_ths = self._copy_array_or_tensor_data(w_max_ths)

            meaningful_properties['weights_thresholds'] = w_min_ths, w_max_ths

        return meaningful_properties
