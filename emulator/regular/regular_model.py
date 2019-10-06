__all__ = [
    'RegularModel',
]

import logging
from typing import List

import tensorflow as tf

from emulator.base_model import BaseModel
from emulator.common.cfg_utils import configuration_has_quantized_layers
from emulator.layers import BaseGraphLayer
from emulator.regular.regular_meta_layer import RegularMetaLayer

_LOGGER = logging.getLogger('emulator.model')


class RegularModel(BaseModel):

    def _check_configuration(self) -> None:
        if configuration_has_quantized_layers(self._cfg):
            raise ValueError(
                'Regular network does not accept configuration data with quantization parameters.'
            )

    @property
    def graph_inputs(self) -> List[tf.Tensor]:
        return [
            self._layers[layer_name].raw_output
            for layer_name in self._input_layers_names
        ]

    @staticmethod
    def _create_layer(layer_inputs: List[BaseGraphLayer], layer_cfg: dict) -> BaseGraphLayer:
        return RegularMetaLayer.create_layer(inputs=layer_inputs, **layer_cfg)
