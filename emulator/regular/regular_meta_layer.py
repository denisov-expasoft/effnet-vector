__all__ = [
    'RegularMetaLayer',
]

import tensorflow as tf

from emulator.common import LayerConfigurationParameters as Lcp
from emulator.layers import BaseMetaLayer
from emulator.layers import LAYERS_REGISTRY

_SUPPORTED_ACTIVATIONS = (None, 'relu', 'swish', 'sigmoid')


class RegularMetaLayer(BaseMetaLayer):

    def __init__(self, **layer_cfg):
        layer_cfg.pop(Lcp.ATTR_INPUT_SHAPE.value, None)
        layer_cfg.pop(Lcp.ATTR_OUTPUT_SHAPE.value, None)

        self._activation_function = layer_cfg.pop(Lcp.ATTR_NN_ACTIVATION_TYPE.value, None)
        if self._activation_function not in _SUPPORTED_ACTIVATIONS:
            raise TypeError(
                f'Unsupported value for the activation function: {self._activation_function}'
            )

        super().__init__(layers_registry=LAYERS_REGISTRY, **layer_cfg)

    def _process_output_tensor(self, output_tensor: tf.Tensor) -> tf.Tensor:
        if self._activation_function is None:
            return output_tensor

        if self._activation_function == 'relu':
            return tf.nn.relu(output_tensor)

        if self._activation_function == 'sigmoid':
            return tf.nn.sigmoid(output_tensor)

        if self._activation_function == 'swish':
            return tf.nn.swish(output_tensor)

        raise NotImplementedError(f'Behaviour of "{self._activation_function}" is not implemented')
