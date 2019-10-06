__all__ = [
    'INTEGER_LAYERS_REGISTRY'
]

from abc import abstractmethod
from copy import deepcopy
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import tensorflow as tf

from emulator.common import Registry
from emulator.common import SupportedLayerTypes as Slt
from emulator.common import create_fixedpoint_scale
from emulator.common.data_types import TBias
from emulator.common.data_types import TWeights
from emulator.common.rounding_registry import NP_ROUNDING_REGISTRY
from emulator.common.rounding_registry import TF_ROUNDING_REGISTRY
from emulator.layers import BackendProxyGraphLayer
from emulator.layers import BaseGraphLayer

INTEGER_LAYERS_REGISTRY = Registry(key_type=Slt)
_TFixedPointScale = Tuple[np.ndarray, np.ndarray]
_SUPPORTED_ACTIVATIONS = (None, 'relu')


class ScalarQuantizationParameters(NamedTuple):
    min_value: float
    max_value: float
    quant_scale: float
    quant_zero: int


class VectorQauntizationParameters(NamedTuple):
    min_value: np.ndarray
    max_value: np.ndarray
    quant_scale: np.ndarray
    quant_zero: np.ndarray


def _tf_round_half_up(data: tf.Tensor) -> tf.Tensor:
    return TF_ROUNDING_REGISTRY['half-up'](data)


def _tf_round_half_away(data: tf.Tensor) -> tf.Tensor:
    return TF_ROUNDING_REGISTRY['half-away'](data)


def _np_round_half_up(data: np.ndarray) -> np.ndarray:
    return NP_ROUNDING_REGISTRY['half-up'](data)


def _np_round_half_away(data: np.ndarray) -> np.ndarray:
    return NP_ROUNDING_REGISTRY['half-away'](data)


class IntegerLayer(BaseGraphLayer):
    def __init__(self, fixed_number_of_inputs: int):
        self._raw_output = None
        super().__init__(fixed_number_of_inputs=fixed_number_of_inputs)

    @property
    def raw_output(self) -> tf.Tensor:
        print('Trying to get the raw output')
        if self._raw_output is not None:
            return self._raw_output
        return super().raw_output


@INTEGER_LAYERS_REGISTRY.add_item_decorator(Slt.LAYER_INPUT)
class InputNode(IntegerLayer):

    def __init__(
            self,
            shape: List[int],
            output_quant_data: ScalarQuantizationParameters,
            dtype: Union[np.dtype, tf.DType, str] = tf.float32,
    ):
        self._shape = shape
        self._dtype = dtype
        self._output_quant_data = deepcopy(output_quant_data)
        print(self._output_quant_data.quant_scale)
        super().__init__(fixed_number_of_inputs=0)

    def _create_backend_operations(self) -> tf.Tensor:

        x = tf.placeholder(self._dtype, self._shape)
        self._raw_output = x

        with tf.name_scope('clip'):
            x = tf.maximum(x, self._output_quant_data.min_value)
            x = tf.minimum(x, self._output_quant_data.max_value)

        with tf.name_scope('quantize'):
            x = tf.subtract(x, self._output_quant_data.min_value)
            x = tf.multiply(x, self._output_quant_data.quant_scale)
            x = _tf_round_half_up(x)

        return x


@INTEGER_LAYERS_REGISTRY.add_item_decorator(Slt.LAYER_OUTPUT)
class OutputNode(BackendProxyGraphLayer):

    def __init__(self):
        super().__init__(
            backend_node_operation=tf.identity,
            fixed_number_of_inputs=1,
        )


class IntegerLayerWithWeights(IntegerLayer):

    def __init__(
            self,
            weights: TWeights,
            bias: TBias,
            input_quant_data: ScalarQuantizationParameters,
            weights_quant_data: VectorQauntizationParameters,
            output_quant_data: ScalarQuantizationParameters,
            activation: Optional[str],
    ):
        self._weights = weights
        self._bias = bias

        self._input_quant_data = deepcopy(input_quant_data)
        self._weights_quant_data = deepcopy(weights_quant_data)
        self._output_quant_data = deepcopy(output_quant_data)

        self._weights_quantized = self._quantize_weights()
        self._bias_quantized = self._quantize_bias()

        if activation not in _SUPPORTED_ACTIVATIONS:
            raise TypeError(
                f'Unsupported value for the activation function: {activation}'
            )

        self._activation = activation

        super().__init__(fixed_number_of_inputs=1)

    def _quantize_weights(self) -> np.ndarray:
        x = np.clip(self._weights, self._weights_quant_data.min_value, self._weights_quant_data.max_value)
        x = x - self._weights_quant_data.min_value
        x = x * self._weights_quant_data.quant_scale
        x = _np_round_half_up(x)
        return x

    def _quantize_bias(self) -> Optional[np.ndarray]:
        if self._bias is None:
            return None
        bias_scale = self._weights_quant_data.quant_scale * self._input_quant_data.quant_scale
        x = self._bias * bias_scale
        x = _np_round_half_away(x)
        return x

    @abstractmethod
    def _create_main_node(self, input_node: tf.Tensor, weights_node: tf.Tensor) -> tf.Tensor:
        pass

    def _create_backend_operations(self) -> tf.Tensor:
        with tf.name_scope('weights'):
            weights_node = self.maybe_save_const(self._weights_quantized, dtype=tf.uint8, name='weights')
            weights_node = tf.cast(weights_node, tf.float64)
            weights_node = tf.subtract(weights_node, self._weights_quant_data.quant_zero, name='fixed_zero')

        bias_node = self.maybe_save_const(self._bias_quantized, dtype=tf.float64, name='bias')

        with tf.name_scope('shift_zero'):
            input_node = self._get_backend_inputs()[0]
            input_node = tf.subtract(input_node, self._input_quant_data.quant_zero)
            input_node = tf.cast(input_node, tf.float64)

        x = self._create_main_node(input_node, weights_node)
        x = tf.nn.bias_add(x, bias_node)

        with tf.name_scope('fixed_point_rescale'):
            # TODO: implement requantization using fixed-point arithmetic

            op_output_scale = self._input_quant_data.quant_scale * self._weights_quant_data.quant_scale
            rescale_factor = self._output_quant_data.quant_scale / op_output_scale

            scale, shift = create_fixedpoint_scale(rescale_factor, 24)
            fp_using_float = scale * 2. ** shift
            fp_using_float = tf.constant(fp_using_float, tf.float64)

            x = tf.multiply(x, fp_using_float)
            x = _tf_round_half_away(x)

            x = tf.cast(x, tf.float32)

        if self._activation is not None:
            with tf.name_scope('activate'):
                x = tf.nn.relu(x)

        with tf.name_scope('to_uint8'):
            output_zero = self.maybe_save_const(
                np.array(self._output_quant_data.quant_zero),
                dtype=tf.uint8,
                name='output_zero',
            )
            output_zero = tf.cast(output_zero, tf.float32)
            x = tf.add(x, output_zero)

        return x


@INTEGER_LAYERS_REGISTRY.add_item_decorator(Slt.LAYER_FC)
class FullyConnected(IntegerLayerWithWeights):

    def _create_main_node(self, input_node: tf.Tensor, weights_node: tf.Tensor) -> tf.Tensor:
        return tf.matmul(input_node, weights_node)


@INTEGER_LAYERS_REGISTRY.add_item_decorator(Slt.LAYER_CONV2D)
class Conv2D(IntegerLayerWithWeights):

    def __init__(
            self,
            strides: List[int],
            padding: str,
            **kwargs,
    ):
        self._strides = strides
        self._padding = padding

        super().__init__(**kwargs)

    def _create_main_node(self, input_node: tf.Tensor, weights_node: tf.Tensor) -> tf.Tensor:
        return tf.nn.conv2d(
            input_node,
            filter=weights_node,
            strides=self._strides,
            padding=self._padding,
        )


@INTEGER_LAYERS_REGISTRY.add_item_decorator(Slt.LAYER_CONV2D_DEPTHWISE)
class DepthwiseConv2D(Conv2D):

    def _create_main_node(self, input_node: tf.Tensor, weights_node: tf.Tensor) -> tf.Tensor:
        return tf.nn.depthwise_conv2d_native(
            input_node,
            filter=weights_node,
            strides=self._strides,
            padding=self._padding,
        )

    def _quantize_weights(self) -> np.ndarray:
        x = np.clip(self._weights, self._weights_quant_data.min_value, self._weights_quant_data.max_value)
        x = x - self._weights_quant_data.min_value
        x = x * np.expand_dims(self._weights_quant_data.quant_scale, -1)
        x = _np_round_half_up(x)
        return x


@INTEGER_LAYERS_REGISTRY.add_item_decorator(Slt.LAYER_REDUCE_MEAN)
class ReduceMean(BackendProxyGraphLayer):

    def __init__(self, axis, keepdims):
        self._axis = axis
        self._keepdims = keepdims
        super().__init__(
            backend_node_operation=tf.reduce_mean,
            fixed_number_of_inputs=1,
            axis=axis,
            keepdims=keepdims,
        )


@INTEGER_LAYERS_REGISTRY.add_item_decorator(Slt.LAYER_ADD)
class AddOperation(BackendProxyGraphLayer):

    def __init__(self):
        super().__init__(backend_node_operation=tf.add, fixed_number_of_inputs=2)


class ReluActivationLayer(BackendProxyGraphLayer):

    def __init__(self):
        super().__init__(
            backend_node_operation=tf.nn.relu,
            fixed_number_of_inputs=1,
        )
