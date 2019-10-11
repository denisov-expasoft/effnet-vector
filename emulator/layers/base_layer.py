__all__ = [
    'BaseGraphLayer',
    'CustomBackendGraphLayer',
    'BackendProxyGraphLayer',
    'BaseGraphLayerWithWeightsAndBias',
    'BasePredefinedNode',
]

from abc import ABCMeta
from abc import abstractmethod
from copy import deepcopy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import tensorflow as tf

from emulator.common import GraphLayerError
from emulator.common import normalize_to_list
from emulator.common.data_types import TArrayOrTensor
from emulator.common.data_types import TBias
from emulator.common.data_types import TWeights


class BaseGraphLayer(metaclass=ABCMeta):

    @staticmethod
    def maybe_save_const(value: TArrayOrTensor, **const_params):
        if isinstance(value, tf.Tensor):
            return value

        return tf.constant(
            value=value,
            **const_params,
        )

    def __init__(self, fixed_number_of_inputs: int = None):
        self._inputs: Optional[List['BaseGraphLayer']] = None
        self._backend_output: Optional[tf.Tensor] = None
        self._fixed_number_of_inputs = fixed_number_of_inputs

    @classmethod
    def create_layer(cls, inputs: Union['BaseGraphLayer', List['BaseGraphLayer']] = None, **kwargs):
        instance = cls(**kwargs)
        return instance(inputs)

    @property
    def inputs(self) -> List['BaseGraphLayer']:
        return self._inputs

    @property
    def backend_output(self) -> tf.Tensor:
        return self._backend_output

    @property
    def raw_output(self) -> tf.Tensor:
        return self.backend_output

    @property
    def output_shape(self) -> Optional[Tuple[int, ...]]:
        if self._backend_output is None:
            return None

        out_shape = self._backend_output.shape.as_list()
        return tuple(out_shape)

    def _check_inputs(self, inputs: List['BaseGraphLayer']):
        if self._fixed_number_of_inputs is None:
            return

        number_of_inputs = len(inputs)
        if self._fixed_number_of_inputs != number_of_inputs:
            raise GraphLayerError(
                f'number of inputs must be {self._fixed_number_of_inputs}, '
                f'but got {number_of_inputs}'
            )

    def _get_backend_inputs(self) -> List[tf.Tensor]:
        return [
            input_item.backend_output
            for input_item in self._inputs
        ]

    @abstractmethod
    def _create_backend_operations(self) -> tf.Tensor:
        pass

    def get_meaningful_properties(self) -> Dict[str, Any]:
        return dict()

    def __call__(self, inputs: Union['BaseGraphLayer', List['BaseGraphLayer']] = None) -> 'BaseGraphLayer':
        inputs = normalize_to_list(inputs)
        self._check_inputs(inputs)

        self._inputs = inputs
        self._backend_output = self._create_backend_operations()

        return self

    @staticmethod
    def _copy_array_or_tensor_data(data: Optional[TArrayOrTensor]) -> Optional[np.ndarray]:
        if data is None:
            return None

        if isinstance(data, (tf.Tensor, tf.Variable)):
            return data.eval()

        return deepcopy(data)


class CustomBackendGraphLayer(BaseGraphLayer):
    """Template for parametric nodes"""

    def __init__(
            self,
            fixed_number_of_inputs: int = None,
            **backend_layer_params,
    ):
        self._backend_layer_params = backend_layer_params

        super().__init__(fixed_number_of_inputs)


class BackendProxyGraphLayer(CustomBackendGraphLayer):
    """Template for simple nodes based on a single operation"""

    def __init__(
            self,
            backend_node_operation,
            fixed_number_of_inputs: int = None,
            **backend_layer_params,
    ):
        self._backend_node_class = backend_node_operation

        super().__init__(fixed_number_of_inputs, **backend_layer_params)

    def _create_backend_operations(self) -> tf.Tensor:
        backend_inputs = self._get_backend_inputs()
        return self._backend_node_class(
            *backend_inputs,
            **self._backend_layer_params,
        )


class BaseGraphLayerWithWeightsAndBias(BaseGraphLayer):

    def __init__(self, weights: TWeights, bias: TBias = None, fixed_number_of_inputs: int = None):
        self._weights = weights
        self._bias = bias
        super().__init__(fixed_number_of_inputs)

    @property
    def has_bias(self) -> bool:
        return self._bias is not None

    @abstractmethod
    def _create_main_node(self, weights_node: tf.Tensor) -> tf.Tensor:
        pass

    def _create_backend_operations(self) -> tf.Tensor:
        weights_node = self.maybe_save_const(self._weights, name='weights')
        backend_output = self._create_main_node(weights_node)

        if self.has_bias:
            bias_node = self.maybe_save_const(self._bias, name='bias')
            backend_output = tf.nn.bias_add(backend_output, bias_node)

        return backend_output


class BasePredefinedNode(BaseGraphLayer):
    def __init__(self, tensor: tf.Tensor):
        self._predefined_tensor = tensor
        super().__init__(fixed_number_of_inputs=0)

    def _create_backend_operations(self) -> tf.Tensor:
        return self._predefined_tensor
