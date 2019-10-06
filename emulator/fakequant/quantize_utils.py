__all__ = [
    'quantize_array_v2',
    'quantize_tensor_v2',
    'nudge_parameters_np',
    'nudge_parameters_tf',
    'discretize_array_or_tensor_v2',
    'create_adjusted_thresholds',
    'create_adjusted_weights',
]

from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import tensorflow as tf

from emulator.common.rounding_registry import NP_ROUNDING_REGISTRY
from emulator.common.rounding_registry import TF_ROUNDING_REGISTRY

_TArrayOrTensor = Union[np.ndarray, tf.Tensor]
_TOptionalArrayOrTensor = Optional[Union[np.ndarray, tf.Tensor]]
_TQuantizedArrayPair = Tuple[np.ndarray, Optional[np.ndarray]]
_TQuantizedTensorPair = Tuple[tf.Tensor, Optional[tf.Tensor]]

_TTensorPair = Tuple[tf.Tensor, tf.Tensor]
_TTensorTriplet = Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
_TArrayPair = Tuple[np.ndarray, np.ndarray]
_TArrayTriplet = Tuple[np.ndarray, np.ndarray, np.ndarray]

_TF_HALF_UP_ROUNDING_FN = TF_ROUNDING_REGISTRY['half-up']
_TF_HALF_AWAY_ROUNDING_FN = TF_ROUNDING_REGISTRY['half-away']
_NP_HALF_UP_ROUNDING_FN = NP_ROUNDING_REGISTRY['half-up']
_NP_HALF_AWAY_ROUNDING_FN = NP_ROUNDING_REGISTRY['half-away']

DEFAULT_THRESHOLDS_MIN_FACTOR = 0.5
DEFAULT_THRESHOLDS_MAX_FACTOR = 1.3
DEFAULT_WEIGHTS_SCALE_MIN_FACTOR = 0.75
DEFAULT_WEIGHTS_SCALE_MAX_FACTOR = 1.25

_DEFAULT_THRESHOLDS_EPS = 1e-10


def _is_same_type_or_none(type_to_check: type, *parameters, **optional_parameters) -> bool:
    if not all(
        isinstance(param, type_to_check)
        for param in parameters
    ):
        return False

    return all(
        isinstance(optional_param, type_to_check) or optional_param is None
        for optional_param in optional_parameters.values()
    )


def _is_bits_acceptable(bits: int) -> bool:
    return 1 < bits <= 32


def bits_to_quant_range(bits: int, narrow_range: bool) -> Tuple[float, float]:
    if not isinstance(bits, int):
        raise TypeError('Bits capacity must be specified via an integer number')

    if not _is_bits_acceptable(bits):
        raise ValueError(
            'Bits capacity must be greater than 1 and less than or equal to 32'
        )

    qmax = 2. ** bits - 1.
    qmin = 1. if narrow_range else 0.

    return qmin, qmax


def maybe_create_constant_tensor(data: Union[np.ndarray, tf.Tensor], **kwargs) -> tf.Tensor:
    if isinstance(data, tf.Tensor):
        return data

    return tf.constant(data, **kwargs)


def quantize_array_v2(
        data: np.ndarray,
        scale: np.ndarray,
        nudged_min: Optional[np.ndarray],
        nudged_max: Optional[np.ndarray],
        dequantize: bool = False,
) -> _TQuantizedArrayPair:
    """Quantization of numpy arrays"""

    boundaries = nudged_min is not None, nudged_max is not None

    if all(boundaries):
        is_clamped = True
    elif any(boundaries):
        raise ValueError('Both boundaries (min / max) must be specified')
    else:
        is_clamped = False

    quantized_data = data

    if is_clamped:
        # In TFLite implementation clipping is used for uint8 quantization
        # It also uses half-away rounding. However, as data before rounding is
        # larger than zero, we can use a simplier half-up rounding instead.
        quantized_data = np.clip(quantized_data, nudged_min, nudged_max)
        quantized_data = (quantized_data - nudged_min) * scale
        quantized_data = _NP_HALF_UP_ROUNDING_FN(quantized_data)

    else:
        # Quantization without preliminary clipping currently is applied
        # to biases only. The data is quantized to int32 values and needs half-away rounding.
        quantized_data = quantized_data * scale
        quantized_data = _NP_HALF_AWAY_ROUNDING_FN(quantized_data)

    # dequantizing
    if dequantize:
        rev_scale = np.reciprocal(scale)

        dequantized_data = quantized_data * rev_scale
        if is_clamped:
            dequantized_data = dequantized_data + nudged_min

    else:
        dequantized_data = None

    return quantized_data, dequantized_data


def quantize_tensor_v2(
        data: tf.Tensor,
        scale: tf.Tensor,
        nudged_min: Optional[tf.Tensor],
        nudged_max: Optional[tf.Tensor],
        dequantize: bool = False,
) -> _TQuantizedTensorPair:
    """Quantization for TensorFlow tensors"""

    boundaries = nudged_min is not None, nudged_max is not None

    if all(boundaries):
        is_clamped = True
    elif any(boundaries):
        raise ValueError('Both boundaries (min / max) must be specified')
    else:
        is_clamped = False

    quantized_data = data

    if is_clamped:
        # In TFLite implementation clipping is used for uint8 quantization
        # It also uses half-away rounding. However, as data before rounding is
        # larger than zero, we can use a simplier half-up rounding instead.
        quantized_data = tf.clip_by_value(quantized_data, nudged_min, nudged_max)
        quantized_data = tf.subtract(quantized_data, nudged_min, name='shifted')
        quantized_data = tf.multiply(quantized_data, scale, name='scaling')
        quantized_data = _TF_HALF_UP_ROUNDING_FN(quantized_data)

    else:
        # Quantization without preliminary clipping currently is applied
        # to biases only. The data is quantized to int32 values and needs half-away rounding.
        quantized_data = tf.multiply(quantized_data, scale, name='scaling')
        quantized_data = _TF_HALF_AWAY_ROUNDING_FN(quantized_data)

    # dequantizing
    if dequantize:
        with tf.name_scope('dequantize'):
            rev_scale = tf.reciprocal(scale, 'reversed_scale')

            dequantized_data = tf.multiply(quantized_data, rev_scale, 'rescaling')
            if is_clamped:
                dequantized_data = tf.add(dequantized_data, nudged_min, name='shift_back')
    else:
        dequantized_data = None

    return quantized_data, dequantized_data


def discretize_array_or_tensor_v2(
        data: _TArrayOrTensor,
        scale: _TArrayOrTensor,
        nudged_min: _TOptionalArrayOrTensor,
        nudged_max: _TOptionalArrayOrTensor,
) -> Union[_TQuantizedArrayPair, _TQuantizedTensorPair]:
    if _is_same_type_or_none(tf.Tensor, data, scale, nudged_min=nudged_min, nudged_max=nudged_max):
        return quantize_tensor_v2(data, scale, nudged_min, nudged_max, dequantize=True)

    if _is_same_type_or_none(np.ndarray, data, scale, nudged_min=nudged_min, nudged_max=nudged_max):
        return quantize_array_v2(data, scale, nudged_min, nudged_max, dequantize=True)

    raise TypeError('All parameters must be of the same type (np.array or tf.Tensor)')


def _create_adjustable_threshold(
        threshold: np.ndarray,
        min_factor: float,
        max_factor: float,
        name: str,
) -> tf.Tensor:
    fixed_thresholds = threshold == 0

    initial_threshold = tf.constant(threshold, tf.float32, name=f'initial_{name}')

    if np.all(fixed_thresholds):
        return initial_threshold

    any_fixed = np.any(fixed_thresholds)

    with tf.name_scope(name):

        variable_name_scope = tf.get_default_graph().get_name_scope()
        with tf.variable_scope(variable_name_scope):
            thresholds_factor = tf.get_variable(
                name='thresholds_factor',
                shape=initial_threshold.shape.as_list(),
                dtype=tf.float32,
                initializer=tf.ones_initializer(dtype=tf.float32),
                trainable=True,
            )

        constrained_thresholds_factor = tf.clip_by_value(
            thresholds_factor,
            clip_value_min=min_factor,
            clip_value_max=max_factor,
            name='constrained_factor',
        )

        adjusted_thresholds = tf.multiply(
            initial_threshold,
            constrained_thresholds_factor,
            name='adjusted_thresholds',
        )

        if not any_fixed:
            return adjusted_thresholds

        fixed_thresholds = tf.constant(fixed_thresholds, tf.bool, name='fixed_thresholds_mask')

        resulting_thresholds = tf.where(
            fixed_thresholds,
            initial_threshold,
            adjusted_thresholds,
            name="filter_fixed_thresholds",
        )

    return resulting_thresholds


def create_adjusted_thresholds(
        min_thresholds: np.ndarray,
        max_thresholds: np.ndarray,
        thresholds_min_factor: float = DEFAULT_THRESHOLDS_MIN_FACTOR,
        thresholds_max_factor: float = DEFAULT_THRESHOLDS_MAX_FACTOR,
) -> _TTensorPair:
    """Create a pair of adjusted thresholds: (min_ths, max_ths, ths_width)"""

    min_thresholds_tensor = _create_adjustable_threshold(
        min_thresholds,
        thresholds_min_factor,
        thresholds_max_factor,
        name='min_thresholds',
    )

    max_thresholds_tensor = _create_adjustable_threshold(
        max_thresholds,
        thresholds_min_factor,
        thresholds_max_factor,
        name='max_thresholds',
    )

    return min_thresholds_tensor, max_thresholds_tensor


def create_adjusted_weights(
        weights: Union[np.ndarray, tf.Tensor],
        min_factor_value: float = DEFAULT_WEIGHTS_SCALE_MIN_FACTOR,
        max_factor_value: float = DEFAULT_WEIGHTS_SCALE_MAX_FACTOR,
) -> tf.Tensor:
    """Create weights with a point-wise scale factor"""

    if isinstance(weights, np.ndarray):
        initial_weights = tf.constant(
            value=weights,
            dtype=tf.float32,
            name='initial_value',
        )

        weights_shape = weights.shape
    else:
        initial_weights = weights
        weights_shape = weights.shape.as_list()

    variable_name_scope = tf.get_default_graph().get_name_scope()
    with tf.variable_scope(variable_name_scope):
        weights_scale_factor = tf.get_variable(
            name='scale_factor',
            shape=weights_shape,
            dtype=tf.float32,
            initializer=tf.ones_initializer(dtype=tf.float32),
            trainable=True,
        )

    weights_scale_factor = tf.clip_by_value(
        weights_scale_factor,
        clip_value_min=min_factor_value,
        clip_value_max=max_factor_value,
        name='constrained_scale_factor',
    )

    adjusted_weights = tf.multiply(
        initial_weights,
        weights_scale_factor,
        name='adjusted_weights',
    )

    return adjusted_weights


def nudge_parameters_tf(
        min_thresholds: tf.Tensor,
        max_thresholds: tf.Tensor,
        bits: int,
        narrow_range: bool,
        thresholds_eps: float = _DEFAULT_THRESHOLDS_EPS,
) -> _TTensorTriplet:

    thresholds_width = tf.subtract(max_thresholds, min_thresholds, 'thresholds_width')

    qmin, qmax = bits_to_quant_range(bits, narrow_range)

    qmin = tf.constant(qmin, shape=thresholds_width.shape, dtype=thresholds_width.dtype, name='q_min')
    qmax = tf.constant(qmax, shape=thresholds_width.shape, dtype=thresholds_width.dtype, name='q_max')

    # Nudging doesn't affect the scale, so we can calculate it using the initial thresholds.
    # We also ensure that distance between thresholds is not less some predefined safe value.
    # (Otherwise it could lead to some numerical issues, especially during the training process,
    # which results in yielding NaN values).
    ths_eps = tf.constant(thresholds_eps, dtype=thresholds_width.dtype, name='ths_eps')
    fixed_thresholds_width = tf.maximum(thresholds_width, ths_eps, name='ensure_non_zero_width')

    scale = tf.truediv(qmax - qmin, fixed_thresholds_width, name='scale')
    reversed_scale = tf.reciprocal(scale, 'reversed_scale')

    # Nudge thresholds according to "tensorflow/compiler/tf2xla/kernels/fake_quantize_ops.cc"
    with tf.name_scope('nudged_zero_point'):
        zero_point_from_min = tf.multiply(-min_thresholds, scale, name='zero_point_from_min')
        rounded_zp = _TF_HALF_UP_ROUNDING_FN(zero_point_from_min, 'rounding_zero_point')

        are_ths_non_negative = tf.less_equal(zero_point_from_min, qmin, 'are_ths_non_negative')
        are_ths_non_positive = tf.greater_equal(zero_point_from_min, qmax, 'are_ths_non_positive')

        nudged_zerop_point = tf.where(
            condition=are_ths_non_negative,
            x=qmin,
            y=tf.where(are_ths_non_positive, qmax, rounded_zp),
        )

    nudged_min = tf.multiply(qmin - nudged_zerop_point, reversed_scale, name='nudged_min')
    nudged_max = tf.multiply(qmax - nudged_zerop_point, reversed_scale, name='nudged_max')

    return nudged_min, nudged_max, scale


def nudge_parameters_np(
        min_thresholds: np.ndarray,
        max_thresholds: np.ndarray,
        bits: int,
        narrow_range: bool,
        thresholds_eps: float = _DEFAULT_THRESHOLDS_EPS,
) -> _TArrayTriplet:

    thresholds_width = max_thresholds - min_thresholds

    qmin, qmax = bits_to_quant_range(bits, narrow_range)

    # Nudging doesn't affect the scale, so we can calculate it using the initial thresholds.
    # We also ensure that distance between thresholds is not less some predefined safe value.
    # (Otherwise it could lead to some numerical issues, especially during the training process,
    # which results in yielding NaN values).
    scale = (qmax - qmin) / np.maximum(thresholds_width, thresholds_eps)
    reversed_scale = np.reciprocal(scale)

    # Nudge thresholds according to "tensorflow/compiler/tf2xla/kernels/fake_quantize_ops.cc"
    zero_point_from_min = -min_thresholds * scale

    nudged_zero_point = np.select(
        condlist=[zero_point_from_min >= qmax, zero_point_from_min > qmin],
        choicelist=[qmax, _NP_HALF_UP_ROUNDING_FN(zero_point_from_min)],
        default=qmin,
    )
    nudged_zero_point = np.asarray(nudged_zero_point, np.float32)

    nudged_min = (qmin - nudged_zero_point) * reversed_scale
    nudged_max = (qmax - nudged_zero_point) * reversed_scale

    return nudged_min, nudged_max, scale


def nudge_parameters_np_ex(
        min_thresholds: np.ndarray,
        max_thresholds: np.ndarray,
        bits: int,
        narrow_range: bool,
        thresholds_eps: float = _DEFAULT_THRESHOLDS_EPS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    thresholds_width = max_thresholds - min_thresholds

    qmin, qmax = bits_to_quant_range(bits, narrow_range)

    # Nudging doesn't affect the scale, so we can calculate it using the initial thresholds.
    # We also ensure that distance between thresholds is not less some predefined safe value.
    # (Otherwise it could lead to some numerical issues, especially during the training process,
    # which results in yielding NaN values).
    scale = (qmax - qmin) / np.maximum(thresholds_width, thresholds_eps)
    reversed_scale = np.reciprocal(scale)

    # Nudge thresholds according to "tensorflow/compiler/tf2xla/kernels/fake_quantize_ops.cc"
    zero_point_from_min = -min_thresholds * scale

    nudged_zero_point = np.select(
        condlist=[zero_point_from_min >= qmax, zero_point_from_min > qmin],
        choicelist=[qmax, _NP_HALF_UP_ROUNDING_FN(zero_point_from_min)],
        default=qmin,
    )
    nudged_zero_point = np.asarray(nudged_zero_point, np.float32)

    nudged_min = (qmin - nudged_zero_point) * reversed_scale
    nudged_max = (qmax - nudged_zero_point) * reversed_scale

    return nudged_min, nudged_max, scale, nudged_zero_point


def nudge_parameters(
        min_thresholds: _TArrayOrTensor,
        max_thresholds: _TArrayOrTensor,
        bits: int,
        narrow_range: bool,
        thresholds_eps: float = _DEFAULT_THRESHOLDS_EPS,
) -> Union[_TTensorTriplet, _TArrayTriplet]:
    if _is_same_type_or_none(tf.Tensor, min_thresholds, max_thresholds):
        return nudge_parameters_tf(min_thresholds, max_thresholds, bits, narrow_range, thresholds_eps)

    if _is_same_type_or_none(np.ndarray, min_thresholds, max_thresholds):
        return nudge_parameters_np(min_thresholds, max_thresholds, bits, narrow_range, thresholds_eps)

    raise TypeError(
        'Both min and max thresholds must have the same type: '
        'np.ndarray or tf.Tensor'
    )
