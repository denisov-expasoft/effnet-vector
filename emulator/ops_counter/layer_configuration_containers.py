from abc import ABC
from abc import abstractproperty
from typing import List

import numpy as np


class BaseOperation(ABC):
    pass

    @abstractproperty
    @property
    def quantized(self) -> bool:
        pass

__all__ = [
    'Conv2D',
    'DepthWiseConv2D',
    'FullyConnected',
    'Add',
    'BaseOperation',
    'GlobalAvg',
    'Sigmoid',
    'Swish',
    'Mul',
    'MatrixOps',
]


class MatrixOps(BaseOperation):
    def __init__(
            self,
            input_size: int,
            kernel_shape: List[int],
            use_bias: bool,
            activation: str,
            weight_bits: int,
            input_bits: List[int],
            output_bits: int,
    ):
        self._input_size = input_size
        self._kernel_shape = kernel_shape
        self._use_bias = use_bias
        self._activation = activation
        self._weight_bits = weight_bits
        self._input_bits = input_bits
        self._output_bits = output_bits

    @property
    def quantized(self) -> bool:
        return self._weight_bits < 16

    @property
    def output_shape(self):
        return [None, self.output_size, self.output_size, self.output_chs_num]

    @property
    def input_size(self) -> int:
        return self._input_size

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
    def input_bits(self) -> List[int]:
        return self._input_bits

    @property
    def output_bits(self) -> int:
        return self._output_bits

    @abstractproperty
    @property
    def output_size(self) -> int:
        pass

    @property
    def input_chs_num(self) -> int:
        return self._kernel_shape[-2]

    @property
    def output_chs_num(self) -> int:
        return self._kernel_shape[-1]


class Conv2D(MatrixOps):
    def __init__(
            self,
            strides: List[int],
            padding: str,
            kernel_shape: List[int],
            **kwargs,
    ):
        if padding not in ['same', 'valid']:
            raise ValueError('Padding must be "same" or "valid"')

        self._strides = strides
        self._padding = padding

        if padding == 'same':
            self._pad = kernel_shape[0] // 2
        else:
            self._pad = 0

        super().__init__(kernel_shape=kernel_shape, **kwargs)

    @property
    def strides(self) -> List[int]:
        return self._strides

    @property
    def padding(self) -> str:
        return self._padding

    @property
    def output_size(self) -> int:
        out_size = np.ceil(
            (self._input_size - self._kernel_shape[0] + 1. + 2 * self._pad) / self._strides[0]
        )
        return int(out_size)


class DepthWiseConv2D(MatrixOps):
    def __init__(
            self,
            strides: List[int],
            padding: str,
            kernel_shape: List[int],
            **kwargs,
    ):
        if padding not in ['same', 'valid']:
            raise ValueError('Padding must be "same" or "valid"')

        self._strides = strides
        self._padding = padding

        if padding == 'same':
            self._pad = kernel_shape[0] // 2
        else:
            self._pad = 0

        super().__init__(kernel_shape=kernel_shape, **kwargs)

    @property
    def strides(self) -> List[int]:
        return self._strides

    @property
    def padding(self) -> str:
        return self._padding

    @property
    def output_chs_num(self) -> int:
        return self._kernel_shape[-2]

    @property
    def output_size(self) -> int:
        out_size = np.ceil(
            (self._input_size - self._kernel_shape[0] + 1. + 2 * self._pad) / self._strides[0]
        )
        return int(out_size)


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
    def quantized(self) -> bool:
        return any(i_bit < 16 for i_bit in self._input_bits)

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
    def quantized(self) -> bool:
        return any(i_bit < 16 for i_bit in self._input_bits)

    @property
    def input_size(self) -> int:
        return self._input_size

    @property
    def n_channels(self) -> int:
        return self._n_channels

    @property
    def input_bits(self) -> List[int]:
        return self._input_bits


class FullyConnected(MatrixOps):
    def __init__(self, **kwargs):
        super().__init__(input_size=1, **kwargs)

    @property
    def output_size(self) -> int:
        return 1

    @property
    def output_shape(self):
        return [None, self.output_chs_num]


class Swish(BaseOperation):

    def __init__(
            self,
            input_shape: List[int],
            input_bits: List[int],
    ):
        self._input_shape = input_shape
        self._input_bits = input_bits

    @property
    def quantized(self) -> bool:
        return any(i_bit < 16 for i_bit in self._input_bits)

    @property
    def input_bits(self) -> List[int]:
        return self._input_bits

    @property
    def input_shape(self) -> List[int]:
        return self._input_shape


class Sigmoid(BaseOperation):

    def __init__(
            self,
            input_shape: List[int],
            input_bits: List[int],
    ):
        self._input_shape = input_shape
        self._input_bits = input_bits

    @property
    def input_bits(self) -> List[int]:
        return self._input_bits

    @property
    def input_shape(self) -> List[int]:
        return self._input_shape

    @property
    def quantized(self) -> bool:
        return any(i_bit < 16 for i_bit in self._input_bits)


class Mul(BaseOperation):

    def __init__(
            self,
            input_shape: List[int],
            input_bits: List[int],
    ):
        self._input_shape = input_shape
        self._input_bits = input_bits

    @property
    def quantized(self) -> bool:
        return any(i_bit < 16 for i_bit in self._input_bits)

    @property
    def input_shape(self) -> List[int]:
        return self._input_shape

    @property
    def input_bits(self) -> List[int]:
        return self._input_bits
