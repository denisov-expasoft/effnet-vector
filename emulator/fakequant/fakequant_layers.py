"""Quantization nodes"""

__all__ = [
    'FakeQuantLayer',
]

from typing import Optional

import tensorflow as tf

from emulator.common.data_types import TArrayOrTensor
from emulator.fakequant.quantize_utils import quantize_tensor_v2
from emulator.layers import BaseGraphLayer


class FakeQuantLayer(BaseGraphLayer):
    """Performs discretization of the input data"""

    def __init__(
            self,
            scale: TArrayOrTensor,
            nudged_min: Optional[TArrayOrTensor],
            nudged_max: Optional[TArrayOrTensor],
    ):
        self._scale = scale
        self._nudged_min = nudged_min
        self._nudged_max = nudged_max

        super().__init__(fixed_number_of_inputs=1)

    def _create_backend_operations(self) -> tf.Tensor:
        backend_inputs = self._get_backend_inputs()
        backend_input = backend_inputs[0]

        scale = self.maybe_save_const(
            self._scale,
            name='fake_quant_scale',
            dtype=backend_input.dtype,
        )

        if self._nudged_min is not None:
            nudged_min = self.maybe_save_const(
                self._nudged_min,
                name='fake_quant_nudged_min',
                dtype=backend_input.dtype,
            )
        else:
            nudged_min = None

        if self._nudged_max is not None:
            nudged_max = self.maybe_save_const(
                self._nudged_max,
                name='fake_quant_nudged_max',
                dtype=backend_input.dtype,
            )
        else:
            nudged_max = None

        _, backend_output = quantize_tensor_v2(
            backend_input,
            scale=scale,
            nudged_min=nudged_min,
            nudged_max=nudged_max,
            dequantize=True,
        )

        return backend_output
