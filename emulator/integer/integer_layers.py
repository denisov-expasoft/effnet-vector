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
_SUPPORTED_ACTIVATIONS = (None, 'relu', 'swish', 'sigmoid')

_FLOAT64 = tf.float64
_FP_BITS = 32

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

        self._weights_quant_zero_point = self._fix_weights_zero_point()
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

    def _fix_weights_zero_point(self):
        return self._weights_quant_data.quant_zero

    @abstractmethod
    def _create_main_node(self, input_node: tf.Tensor, weights_node: tf.Tensor) -> tf.Tensor:
        pass

    def _create_backend_operations(self) -> tf.Tensor:
        with tf.name_scope('weights'):
            weights_node = self.maybe_save_const(self._weights_quantized, dtype=tf.uint8, name='weights')
            weights_node = tf.cast(weights_node, _FLOAT64)
            weights_node = tf.subtract(weights_node, self._weights_quant_zero_point, name='fixed_zero')

        bias_node = self.maybe_save_const(self._bias_quantized, dtype=tf.int32, name='quantized_bias')
        bias_node = tf.cast(bias_node, _FLOAT64, name='bias')

        with tf.name_scope('shift_zero'):
            input_node = self._get_backend_inputs()[0]
            input_node = tf.subtract(input_node, self._input_quant_data.quant_zero)
            input_node = tf.cast(input_node, _FLOAT64)

        x = self._create_main_node(input_node, weights_node)
        x = tf.nn.bias_add(x, bias_node)

        op_output_scale = self._input_quant_data.quant_scale * self._weights_quant_data.quant_scale

        if self._activation in [None, 'relu']:
            with tf.name_scope('fixed_point_rescale'):
                # Standard pipeline with requantization and applying activation function
                rescale_factor = self._output_quant_data.quant_scale / op_output_scale
                scale, shift = create_fixedpoint_scale(rescale_factor, _FP_BITS)
                fp_using_float = scale * 2. ** shift
                fp_using_float = tf.constant(fp_using_float, _FLOAT64)
                # assert False  # TODO: remove

                x = tf.multiply(x, fp_using_float)
                x = tf.cast(x, tf.float32)
                x = _tf_round_half_away(x)

            if self._activation == 'relu':
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

        elif self._activation in ['swish', 'sigmoid']:
        # if self._activation in ['swish', 'sigmoid', None, 'relu']:
            dequantize_factor = 1. / op_output_scale

            with tf.name_scope('dequantize'):
                x = tf.multiply(x, dequantize_factor)
                x = tf.cast(x, tf.float32, 'to_float')

            with tf.name_scope('activate'):
                if self._activation == 'sigmoid':
                    x = tf.nn.sigmoid(x)
                elif self._activation == 'swish':
                    x = tf.nn.swish(x)
                # elif self._activation == 'relu':
                #     x = tf.nn.relu(x)

            with tf.name_scope('clip'):
                x = tf.maximum(x, self._output_quant_data.min_value)
                x = tf.minimum(x, self._output_quant_data.max_value)

            with tf.name_scope('quantize'):
                x = tf.subtract(x, self._output_quant_data.min_value)
                x = tf.multiply(x, self._output_quant_data.quant_scale)
                x = _tf_round_half_up(x)

        else:
            raise NotImplementedError(f'Behaviour of "{self._activation}" is not implemented')

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

    def _fix_weights_zero_point(self):
        return np.expand_dims(self._weights_quant_data.quant_zero, -1)

    def _quantize_weights(self) -> np.ndarray:
        x = np.clip(self._weights, self._weights_quant_data.min_value, self._weights_quant_data.max_value)
        x = x - self._weights_quant_data.min_value
        x = x * np.expand_dims(self._weights_quant_data.quant_scale, -1)
        x = _np_round_half_up(x)
        return x


@INTEGER_LAYERS_REGISTRY.add_item_decorator(Slt.LAYER_REDUCE_MEAN)
class ReduceMean(IntegerLayer):

    def __init__(self, axis, keepdims, output_quant_data: ScalarQuantizationParameters = None):
        self._axis = axis
        self._keepdims = keepdims
        self._output_quant_data = output_quant_data
        super().__init__(fixed_number_of_inputs=1)

    def _create_backend_operations(self) -> tf.Tensor:

        input_node = self._get_backend_inputs()[0]

        input_node_shape = input_node.shape.as_list()
        spatial_dims = [input_node_shape[i] for i in self._axis]
        w_area = int(np.prod(spatial_dims))

        x = input_node
        x = tf.cast(x, _FLOAT64)

        x = tf.reduce_sum(x, axis=self._axis, keepdims=self._keepdims)
        x = tf.math.floordiv(x, w_area)

        return x


@INTEGER_LAYERS_REGISTRY.add_item_decorator(Slt.LAYER_ADD)
class AddOperation(IntegerLayer):

    def __init__(
            self,
            input_1_quant_data: ScalarQuantizationParameters,
            input_2_quant_data: ScalarQuantizationParameters,
            output_quant_data: VectorQauntizationParameters,
    ):
        self._input_1_quant_data = input_1_quant_data
        self._input_2_quant_data = input_2_quant_data
        self._output_quant_data = output_quant_data
        super().__init__(fixed_number_of_inputs=2)

    def _create_backend_operations(self) -> tf.Tensor:
        input_node_1, input_node_2 = self._get_backend_inputs()

        with tf.name_scope('dequantize_1'):
            input_node_1 = tf.subtract(input_node_1, self._input_1_quant_data.quant_zero)
            input_node_1 = tf.multiply(input_node_1, 1. / self._input_1_quant_data.quant_scale)

        with tf.name_scope('dequantize_2'):
            input_node_2 = tf.subtract(input_node_2, self._input_2_quant_data.quant_zero)
            input_node_2 = tf.multiply(input_node_2, 1. / self._input_2_quant_data.quant_scale)

        x = tf.add(input_node_1, input_node_2)

        with tf.name_scope('clip'):
            x = tf.maximum(x, self._output_quant_data.min_value)
            x = tf.minimum(x, self._output_quant_data.max_value)

        with tf.name_scope('quantize'):
            x = tf.subtract(x, self._output_quant_data.min_value)
            x = tf.multiply(x, self._output_quant_data.quant_scale)
            x = _tf_round_half_up(x)

        return x


@INTEGER_LAYERS_REGISTRY.add_item_decorator(Slt.LAYER_MUL)
class MulOperation(IntegerLayer):

    def __init__(
            self,
            input_1_quant_data: ScalarQuantizationParameters,
            input_2_quant_data: ScalarQuantizationParameters,
            output_quant_data: VectorQauntizationParameters,
    ):
        self._input_1_quant_data = input_1_quant_data
        self._input_2_quant_data = input_2_quant_data
        self._output_quant_data = output_quant_data
        super().__init__(fixed_number_of_inputs=2)

    def _create_backend_operations(self) -> tf.Tensor:

        input_node_1, input_node_2 = self._get_backend_inputs()

        with tf.name_scope('shift_zero_1'):
            input_node_1 = tf.subtract(input_node_1, self._input_1_quant_data.quant_zero)

        with tf.name_scope('shift_zero_2'):
            input_node_2 = tf.subtract(input_node_2, self._input_2_quant_data.quant_zero)

        x = tf.multiply(input_node_1, input_node_2)

        op_output_scale = self._input_1_quant_data.quant_scale * self._input_2_quant_data.quant_scale

        with tf.name_scope('dequantize'):
            x = tf.multiply(x, 1. / op_output_scale)

        with tf.name_scope('clip'):
            x = tf.maximum(x, self._output_quant_data.min_value)
            x = tf.minimum(x, self._output_quant_data.max_value)

        with tf.name_scope('quantize'):
            x = tf.subtract(x, self._output_quant_data.min_value)
            x = tf.multiply(x, self._output_quant_data.quant_scale)
            x = _tf_round_half_up(x)

        # with tf.name_scope('fixed_point_rescale'):
        #     # Standard pipeline with requantization and applying activation function
        #     rescale_factor = self._output_quant_data.quant_scale / op_output_scale
        #     scale, shift = create_fixedpoint_scale(rescale_factor, _FP_BITS)
        #     fp_using_float = scale * 2. ** shift
        #     fp_using_float = tf.constant(fp_using_float, _FLOAT64)
        #
        #     x = tf.cast(x, _FLOAT64)
        #     x = tf.multiply(x, fp_using_float)
        #     x = tf.cast(x, tf.float32)
        #     x = _tf_round_half_away(x)
        #
        # with tf.name_scope('to_uint8'):
        #     output_zero = self.maybe_save_const(
        #         np.array(self._output_quant_data.quant_zero),
        #         dtype=tf.uint8,
        #         name='output_zero',
        #     )
        #     output_zero = tf.cast(output_zero, tf.float32)
        #     x = tf.add(x, output_zero)

        return x


# class ReluActivationLayer(BackendProxyGraphLayer):
#
#     def __init__(self):
#         super().__init__(
#             backend_node_operation=tf.nn.relu,
#             fixed_number_of_inputs=1,
#         )
