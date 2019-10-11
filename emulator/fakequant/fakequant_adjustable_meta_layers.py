__all__ = [
    'FAKEQUANT_ADJUSTABLE_THRESHOLDS_METALAYERS',
    'FAKEQUANT_ADJUSTABLE_WEIGHTS_METALAYERS',
    'AdjWeightsMatrixOpsMetaLayer',
]

from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
import tensorflow as tf

from emulator.common import Registry
from emulator.common import SupportedLayerTypes as Slt
from emulator.common import drop_nones
from emulator.common.data_types import TArrayOrTensor
from emulator.fakequant.calibrators import BaseCalibrator
from emulator.fakequant.calibrators import MinMaxCalibrator
from emulator.fakequant.calibrators import WEIGHTS_CALIBRATORS_REGISTRY
from emulator.fakequant.fakequant_meta_layer import FQMatrixOpsMetaLayer
from emulator.fakequant.fakequant_meta_layer import FQMetaLayerWithActivation
from emulator.fakequant.fakequant_meta_layer import FQRequantizationMetaLayer
from emulator.fakequant.quantize_utils import create_adjusted_thresholds
from emulator.fakequant.quantize_utils import create_adjusted_weights
from emulator.fakequant.quantize_utils import nudge_parameters_tf

FAKEQUANT_ADJUSTABLE_THRESHOLDS_METALAYERS = Registry(key_type=Slt)
FAKEQUANT_ADJUSTABLE_WEIGHTS_METALAYERS = Registry(key_type=Slt)


@FAKEQUANT_ADJUSTABLE_THRESHOLDS_METALAYERS.add_item_decorator([
    Slt.LAYER_INPUT,
    Slt.LAYER_ADD,
])
class AdjThsRequantizationMetaLayer(FQRequantizationMetaLayer):
    def __init__(self, **layer_cfg):
        self._activations_adjusted_thresholds = None
        super().__init__(**layer_cfg)

    @property
    def activations_thresholds(self) -> Optional[Tuple[TArrayOrTensor, TArrayOrTensor]]:
        return self._activations_adjusted_thresholds

    def _maybe_calibrate_output(self) -> None:
        if not self._outputs_quantized:
            return

        calibrator = MinMaxCalibrator()

        min_thresholds, max_thresholds = self._activations_thresholds

        with tf.name_scope('output_adjustable_thresholds'):
            adj_min_ths, adj_max_ths = create_adjusted_thresholds(
                min_thresholds=min_thresholds,
                max_thresholds=max_thresholds,
            )

        with tf.name_scope('output_nudged_parameters'):
            nudged_min, nudged_max, scale = nudge_parameters_tf(
                min_thresholds=adj_min_ths,
                max_thresholds=adj_max_ths,
                bits=self._activations_bits,
                narrow_range=False,
            )
            scale = calibrator.to_output_format_tf(scale)

        self._activations_nudged_min = nudged_min
        self._activations_nudged_max = nudged_max
        self._activations_scale = scale
        self._activations_adjusted_thresholds = adj_min_ths, adj_max_ths

        self._output_scale = self._activations_scale


class AdjThsMetaLayerWithActivation(AdjThsRequantizationMetaLayer, FQMetaLayerWithActivation):
    pass


@FAKEQUANT_ADJUSTABLE_THRESHOLDS_METALAYERS.add_item_decorator([
    Slt.LAYER_CONV2D,
    Slt.LAYER_CONV2D_DEPTHWISE,
    Slt.LAYER_FC,
])
class AdjThsMatrixOpsMetaLayer(AdjThsMetaLayerWithActivation, FQMatrixOpsMetaLayer):
    def __init__(self, **layer_cfg):
        self._weights_adjusted_thresholds = None
        super().__init__(**layer_cfg)

    @property
    def weights_thresholds(self) -> Optional[Tuple[TArrayOrTensor, TArrayOrTensor]]:
        return self._weights_adjusted_thresholds

    def _maybe_calibrate_weights(self) -> None:

        if not self._weights_quantized:
            return

        calibrator_class = WEIGHTS_CALIBRATORS_REGISTRY[self._weights_calibration_type]
        calibrator: BaseCalibrator = calibrator_class()

        min_thresholds, max_thresholds = self._weights_thresholds

        with tf.name_scope('weights_adjustable_thresholds'):
            adj_min_ths, adj_max_ths = create_adjusted_thresholds(
                min_thresholds=min_thresholds,
                max_thresholds=max_thresholds,
            )

        with tf.name_scope('weights_nudged_parameters'):
            nudged_min, nudged_max, scale = nudge_parameters_tf(
                min_thresholds=adj_min_ths,
                max_thresholds=adj_max_ths,
                bits=self._weights_bits,
                narrow_range=self._weights_narrow_range,
            )
            transformed_scale = calibrator.to_output_format_tf(scale)

        self._weights_nudged_min = nudged_min
        self._weights_nudged_max = nudged_max

        self._weights_scale = scale
        self._weights_output_scale = transformed_scale

        self._weights_adjusted_thresholds = adj_min_ths, adj_max_ths
        self._bias_scale = self._create_bias_scale()

        self._output_scale = self._bias_scale

    def _create_bias_scale(self) -> TArrayOrTensor:
        with tf.name_scope('bias_scale'):
            return self._input_scale * self._weights_output_scale


@FAKEQUANT_ADJUSTABLE_WEIGHTS_METALAYERS.add_item_decorator([
    Slt.LAYER_INPUT,
    Slt.LAYER_ADD,
])
class AdjWeightsRequantizationMetaLayer(AdjThsRequantizationMetaLayer):
    pass


@FAKEQUANT_ADJUSTABLE_WEIGHTS_METALAYERS.add_item_decorator([
    Slt.LAYER_CONV2D,
    Slt.LAYER_CONV2D_DEPTHWISE,
    Slt.LAYER_FC,
])
class AdjWeightsMatrixOpsMetaLayer(FQMatrixOpsMetaLayer):
    def _maybe_create_adjustable_weights(self) -> None:
        if self._weights is None:
            return

        with tf.name_scope('adjusted_weights'):
            self._weights = create_adjusted_weights(self._weights)

        if self._bias is None:
            return

        with tf.name_scope('adjusted_bias'):
            self._bias = create_adjusted_weights(self._bias)

    def _maybe_calibrate_weights(self) -> None:
        self._maybe_create_adjustable_weights()
        super()._maybe_calibrate_weights()

    def _create_bias_scale(self) -> TArrayOrTensor:
        with tf.name_scope('bias_scale'):
            return self._input_scale * self._weights_output_scale

    def get_weights(self) -> Dict[str, np.ndarray]:
        result = {
            'weights': self._copy_array_or_tensor_data(self._weights),
            'bias': self._copy_array_or_tensor_data(self._bias),
        }
        return drop_nones(result)
