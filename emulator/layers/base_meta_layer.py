__all__ = [
    'BaseMetaLayer',
]

from typing import List
from typing import Optional
from typing import Type
from typing import Union

import tensorflow as tf

from emulator.common import LayerConfigurationParameters as Lcp
from emulator.common import Registry
from emulator.common import SupportedLayerTypes as Slt
from emulator.layers.base_layer import BaseGraphLayer


class BaseMetaLayer(BaseGraphLayer):
    def __init__(self, layers_registry: Registry, **layer_cfg):
        if not isinstance(layers_registry, Registry):
            raise TypeError(
                'layers_registry must be an instance of the `emulator.common.registry.Registry` class'
            )

        self._layers_registry = layers_registry
        layer_class = layer_cfg.pop(Lcp.ATTR_COMMON_TYPE.value)
        layer_class = self._maybe_retrieve_layer_class(layer_class)

        self._layer_class = layer_class
        self._base_layer_cfg = layer_cfg
        self._raw_output_layer: Optional[BaseGraphLayer] = None

        super().__init__(fixed_number_of_inputs=None)

    @property
    def raw_output(self) -> tf.Tensor:
        return self._raw_output_layer.backend_output

    def _maybe_retrieve_layer_class(
            self,
            layer_class: Union[str, Slt, Type[BaseGraphLayer]]
    ) -> Type[BaseGraphLayer]:
        if isinstance(layer_class, str):
            layer_class = Slt(layer_class)
            return self._get_class_from_registry(layer_class)

        if isinstance(layer_class, Slt):
            return self._get_class_from_registry(layer_class)

        if isinstance(layer_class, type) and issubclass(layer_class, BaseGraphLayer):
            return layer_class

        raise ValueError(
            'Specified layer class must be a string or an enum'
        )

    def _get_class_from_registry(self, layer_class_enum: Slt):
        return self._layers_registry[layer_class_enum]

    def _create_layer(self, layer_inputs: List[BaseGraphLayer]) -> BaseGraphLayer:
        layer_cfg = self._prepare_cfg()
        return self._layer_class.create_layer(layer_inputs, **layer_cfg)

    def _prepare_cfg(self) -> dict:
        return self._base_layer_cfg

    def _preprocess_input_layers(self, layer_inputs: List[BaseGraphLayer]) -> List[BaseGraphLayer]:
        return layer_inputs

    def _process_output_layer(self, output_layer: BaseGraphLayer) -> BaseGraphLayer:
        return output_layer

    def _process_output_tensor(self, output_tensor: tf.Tensor) -> tf.Tensor:
        return output_tensor

    def _create_backend_operations(self) -> tf.Tensor:

        layer_inputs = self._preprocess_input_layers(self._inputs)
        self._raw_output_layer = self._create_layer(layer_inputs)
        output_layer = self._process_output_layer(self._raw_output_layer)

        backend_output = output_layer.backend_output
        backend_output = self._process_output_tensor(backend_output)
        backend_output = tf.identity(backend_output, name='output')

        return backend_output
