__all__ = [
    'InputNode',
    'OutputNode',
    'Conv2D',
    'MulOperation',
    'DepthwiseConv2D',
    'ReduceMean',
    'FullyConnected',
    'ReluActivationLayer',
    'LAYERS_REGISTRY',
]

from typing import List
from typing import Union

import numpy as np
import tensorflow as tf

from emulator.common import Registry
from emulator.common import SupportedLayerTypes as Slt
from emulator.common.data_types import TBias
from emulator.common.data_types import TWeights
from emulator.layers.base_layer import BackendProxyGraphLayer
from emulator.layers.base_layer import BaseGraphLayerWithWeightsAndBias

LAYERS_REGISTRY = Registry(key_type=Slt)


@LAYERS_REGISTRY.add_item_decorator(Slt.LAYER_INPUT)
class InputNode(BackendProxyGraphLayer):

    def __init__(
            self,
            shape: List[int],
            anchor: str,
            dtype: Union[np.dtype, tf.DType, str] = tf.float32,
            name: str = None,
    ):
        self._anchor = anchor
        super().__init__(
            backend_node_operation=tf.placeholder,
            fixed_number_of_inputs=0,
            dtype=dtype,
            shape=shape,
            name=name,
        )

    @property
    def anchor(self) -> str:
        return self._anchor


@LAYERS_REGISTRY.add_item_decorator(Slt.LAYER_OUTPUT)
class OutputNode(BackendProxyGraphLayer):

    def __init__(self, anchor: str):
        self._anchor = anchor
        super().__init__(
            backend_node_operation=tf.identity,
            fixed_number_of_inputs=1,
        )

    @property
    def anchor(self) -> str:
        return self._anchor


@LAYERS_REGISTRY.add_item_decorator(Slt.LAYER_CONV2D)
class Conv2D(BaseGraphLayerWithWeightsAndBias):

    def __init__(
            self,
            weights: TWeights,
            bias: TBias = None,
            strides: List[int] = None,
            dilations: List[int] = None,
            padding: str = None,
    ):
        self._strides = strides
        self._padding = padding
        self._dilations = dilations

        super().__init__(
            weights=weights,
            bias=bias,
            fixed_number_of_inputs=1,
        )

    def _create_main_node(self, weights_node):
        backend_inputs = self._get_backend_inputs()
        backend_input = backend_inputs[0]

        backend_output = tf.nn.conv2d(
            backend_input,
            filter=weights_node,
            strides=self._strides,
            dilations=self._dilations,
            padding=self._padding,
        )

        return backend_output


@LAYERS_REGISTRY.add_item_decorator(Slt.LAYER_CONV2D_DEPTHWISE)
class DepthwiseConv2D(Conv2D):

    def _create_main_node(self, weights_node: tf.Tensor) -> tf.Tensor:
        backend_inputs = self._get_backend_inputs()
        backend_output = tf.nn.depthwise_conv2d_native(
            backend_inputs[0],
            filter=weights_node,
            strides=self._strides,
            dilations=self._dilations,
            padding=self._padding,
        )

        return backend_output


@LAYERS_REGISTRY.add_item_decorator(Slt.LAYER_REDUCE_MEAN)
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


@LAYERS_REGISTRY.add_item_decorator(Slt.LAYER_FC)
class FullyConnected(BaseGraphLayerWithWeightsAndBias):

    def __init__(self, weights: TWeights, bias: TBias = None):
        super().__init__(weights, bias, fixed_number_of_inputs=1)

    def _create_main_node(self, weights_node: tf.Tensor) -> tf.Tensor:
        backend_inputs = self._get_backend_inputs()

        backend_output = tf.matmul(
            backend_inputs[0],
            weights_node,
        )

        return backend_output


@LAYERS_REGISTRY.add_item_decorator(Slt.LAYER_ADD)
class AddOperation(BackendProxyGraphLayer):

    def __init__(self):
        super().__init__(backend_node_operation=tf.add, fixed_number_of_inputs=2)


@LAYERS_REGISTRY.add_item_decorator(Slt.LAYER_MUL)
class MulOperation(BackendProxyGraphLayer):

    def __init__(self):
        super().__init__(backend_node_operation=tf.multiply, fixed_number_of_inputs=2)


class ReluActivationLayer(BackendProxyGraphLayer):

    def __init__(self):
        super().__init__(
            backend_node_operation=tf.nn.relu,
            fixed_number_of_inputs=1,
        )
