__all__ = [
    'IntegerMetaLayer',
]

import tensorflow as tf

from emulator.common import LayerConfigurationParameters as Lcp
from emulator.integer.integer_layers import INTEGER_LAYERS_REGISTRY
from emulator.layers import BaseMetaLayer

_SUPPORTED_ACTIVATIONS = (None, 'relu')


class IntegerMetaLayer(BaseMetaLayer):

    def __init__(self, **layer_cfg):

        layer_cfg.pop(Lcp.ATTR_INPUT_SHAPE.value, None)
        layer_cfg.pop(Lcp.ATTR_OUTPUT_SHAPE.value, None)
        layer_cfg.pop(Lcp.ATTR_IO_GRAPH_ANCHOR.value, None)

        super().__init__(layers_registry=INTEGER_LAYERS_REGISTRY, **layer_cfg)

    @property
    def raw_output(self) -> tf.Tensor:
        return self._raw_output_layer.raw_output
