from abc import ABC
from typing import List


class BaseOperation(ABC):
    pass


__all__ = [
    'Conv2D',
    'DepthWiseConv2D',
    'FullyConnected',
    'Add',
    'BaseOperation',
    'GlobalAvg',
    'AvgPool',
    'Sigmoid',
    'Swish',
    'Mul',
]


class Conv2D(BaseOperation):
    def __init__(
            self,
            input_size: int,
            kernel_shape: List[int],
            strides: List[int],
            padding: str,
            use_bias: bool,
            activation: str,
            activation_bits: int,
            weight_bits: int,
            input_bits: List[int],
            output_bits: int,
    ):
        self._input_size = input_size
        self._kernel_shape = kernel_shape
        self._strides = strides
        self._padding = padding
        self._use_bias = use_bias
        self._activation = activation
        self._activation_bits = activation_bits
        self._weight_bits = weight_bits
        self._input_bits = input_bits
        self._output_bits = output_bits

    @property
    def input_size(self) -> int:
        return self._input_size

    @property
    def kernel_shape(self) -> List[int]:
        return self._kernel_shape

    @property
    def strides(self) -> List[int]:
        return self._strides

    @property
    def padding(self) -> str:
        return self._padding

    @property
    def use_bias(self) -> bool:
        return self._use_bias

    @property
    def activation(self) -> str:
        return self._activation

    @property
    def activation_bits(self) -> int:
        return self._activation_bits

    @property
    def weight_bits(self) -> int:
        return self._weight_bits

    @property
    def input_bits(self) -> List[int]:
        return self._input_bits

    @property
    def output_bits(self) -> int:
        return self.output_bits


class DepthWiseConv2D(BaseOperation):
    def __init__(
            self,
            input_size: int,
            kernel_shape: List[int],
            strides: List[int],
            padding: str,
            use_bias: bool,
            activation: str,
            activation_bits: int,
            weight_bits: int,
            input_bits: List[int],
            output_bits: int,
    ):
        self._input_size = input_size
        self._kernel_shape = kernel_shape
        self._strides = strides
        self._padding = padding
        self._use_bias = use_bias
        self._activation = activation
        self._activation_bits = activation_bits
        self._weight_bits = weight_bits
        self._input_bits = input_bits
        self._output_bits = output_bits

    @property
    def input_size(self) -> int:
        return self._input_size

    @property
    def kernel_shape(self) -> List[int]:
        return self._kernel_shape

    @property
    def strides(self) -> List[int]:
        return self._strides

    @property
    def padding(self) -> str:
        return self._padding

    @property
    def use_bias(self) -> bool:
        return self._use_bias

    @property
    def activation(self) -> str:
        return self._activation

    @property
    def activation_bits(self) -> int:
        return self._activation_bits

    @property
    def weight_bits(self) -> int:
        return self._weight_bits

    @property
    def input_bits(self) -> List[int]:
        return self._input_bits

    @property
    def output_bits(self) -> int:
        return  self._output_bits


class Add(BaseOperation):
    def __init__(
            self,
            input_size: int,
            n_channels: int,
            output_bits: int,
            input_bits: List[int],
    ):
        self._input_size = input_size
        self._n_channels = n_channels
        self._output_bits = output_bits
        self._input_bits = input_bits

    @property
    def input_size(self) -> int:
        return self._input_size

    @property
    def n_channels(self) -> int:
        return self._n_channels

    @property
    def input_bits(self) -> List[int]:
        return self._input_bits

    @property
    def output_bits(self) -> int:
        return self._output_bits


class GlobalAvg(BaseOperation):
    def __init__(
            self,
            input_size: int,
            n_channels: int,
            input_bits: List[int],
    ):
        self._input_size = input_size
        self._n_channels = n_channels
        self._input_bits = input_bits

    @property
    def input_size(self) -> int:
        return self._input_size

    @property
    def n_channels(self) -> int:
        return self._n_channels

    @property
    def input_bits(self) -> List[int]:
        return self._input_bits


class FullyConnected(BaseOperation):

    def __init__(
            self,
            kernel_shape: List[int],
            use_bias: bool,
            activation: str,
            weight_bits: int,
            activation_bits: int,
            input_bits: List[int],
            output_bits: int,
    ):
        self._kernel_shape = kernel_shape
        self._use_bias = use_bias
        self._activation = activation
        self._weight_bits = weight_bits
        self._activation_bits = activation_bits
        self._input_bits = input_bits
        self._output_bits = output_bits

    @property
    def kernel_shape(self) -> List[int]:
        return self._kernel_shape

    @property
    def use_bias(self) -> bool:
        return self._use_bias

    @property
    def activation(self) -> str:
        return self._activation

    @property
    def weight_bits(self) -> int:
        return self._weight_bits

    @property
    def activation_bits(self) -> int:
        return self._activation_bits

    @property
    def input_bits(self) -> List[int]:
        return self._input_bits

    @property
    def output_bits(self) -> int:
        return self._output_bits


class AvgPool(BaseOperation):

    def __init__(
            self,
            input_size: int,
            kernel_shape: List[int],
            strides: List[int],
            padding: str,
            input_bits: List[int],
            input_channels: int,
    ):
        self._input_size = input_size
        self._kernel_shape = kernel_shape
        self._strides = strides
        self._padding = padding
        self._input_bits = input_bits
        self._input_channels = input_channels

    @property
    def input_size(self) -> int:
        return self._input_size

    @property
    def kernel_shape(self) -> List[int]:
        return self._kernel_shape

    @property
    def strides(self) -> List[int]:
        return self._strides

    @property
    def padding(self) -> str:
        return self._padding

    @property
    def input_bits(self) -> List[int]:
        return self._input_bits

    @property
    def input_channels(self) -> int:
        return self._input_channels


class Swish(BaseOperation):

    def __init__(
            self,
            input_shape: List[int],
            input_bits: List[int],
            activation: str,
    ):
        self._input_shape = input_shape
        self._input_bits = input_bits
        self._activation = activation

    @property
    def input_bits(self) -> List[int]:
        return self._input_bits

    @property
    def input_shape(self) -> List[int]:
        return self._input_shape

    @property
    def activation(self) -> str:
        return self._activation


class Sigmoid(BaseOperation):

    def __init__(
            self,
            input_shape: List[int],
            input_bits: List[int],
            activation: str,
    ):
        self._input_shape = input_shape
        self._input_bits = input_bits
        self._activation = activation

    @property
    def input_bits(self) -> List[int]:
        return self._input_bits

    @property
    def input_shape(self) -> List[int]:
        return self._input_shape

    @property
    def activation(self) -> str:
        return self._activation


class Mul(BaseOperation):

    def __init__(
            self,
            input_shape: List[int],
            input_bits: List[int],
    ):
        self._input_shape = input_shape
        self._input_bits = input_bits

    @property
    def input_shape(self) -> List[int]:
        return self._input_shape

    @property
    def input_bits(self) -> List[int]:
        return self._input_bits
