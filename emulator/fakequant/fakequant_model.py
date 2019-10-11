__all__ = ['FakeQuantModel']

from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from emulator.common import LayerConfigurationParameters as Lcp
from emulator.common import SupportedLayerTypes as Slt
from emulator.common.cfg_utils import configuration_has_quantized_layers
from emulator.common.data_types import TWeightsCfg
from emulator.common.data_types import TLayerWeightsCfg
from emulator.fakequant.calibrators import ThresholdsMappedData
from emulator.regular import RegularModel
from emulator.layers import BaseGraphLayer
from emulator.fakequant.fakequant_meta_layer import FAKEQUANT_METALAYERS_REGISTRY
from emulator.fakequant.fakequant_meta_layer import FQMetaLayerEnvelope


class FakeQuantModel(RegularModel):
    def __init__(
            self,
            cfg_or_cfg_path: Union[str, Path, dict],
            weights: TWeightsCfg,
            activations_threshold_data: ThresholdsMappedData,
            weights_threshold_data: ThresholdsMappedData,
    ):
        if not isinstance(activations_threshold_data, ThresholdsMappedData):
            raise TypeError(
                'Activations threshold data must be provided via an instance of the ThresholdsMappedData class'
            )

        if not isinstance(weights_threshold_data, ThresholdsMappedData):
            raise TypeError(
                'Weights threshold data must be provided via an instance of the ThresholdsMappedData class'
            )

        self._precalculated_activations_thresholds = activations_threshold_data.thresholds_dict
        self._precalculated_weights_thresholds = weights_threshold_data.thresholds_dict

        super().__init__(cfg_or_cfg_path, weights)

    def _check_configuration(self) -> None:
        if not configuration_has_quantized_layers(self._cfg):
            raise ValueError(
                'FakeQuant model requires configuration data with specified quantization parameters.'
            )

    def _prepare_layer_inputs_and_cfg(
            self,
            layer_name: str,
            layer_weights: TLayerWeightsCfg,
    ) -> Tuple[List[BaseGraphLayer], dict]:
        layer_inputs, layer_cfg = super()._prepare_layer_inputs_and_cfg(layer_name, layer_weights)

        if layer_name in self._precalculated_activations_thresholds:
            layer_cfg['activations_thresholds_data'] = self._precalculated_activations_thresholds[layer_name]

        if layer_name in self._precalculated_weights_thresholds:
            layer_cfg['weights_thresholds_data'] = self._precalculated_weights_thresholds[layer_name]

        return layer_inputs, layer_cfg

    @staticmethod
    def _create_layer(layer_inputs: List[BaseGraphLayer], layer_cfg: dict) -> BaseGraphLayer:
        layer_type = layer_cfg[Lcp.ATTR_COMMON_TYPE.value]
        layer_class = FAKEQUANT_METALAYERS_REGISTRY.get(Slt(layer_type), FQMetaLayerEnvelope)
        return layer_class.create_layer(inputs=layer_inputs, **layer_cfg)

    def get_network_thresholds(self) -> Optional[Tuple[ThresholdsMappedData, ThresholdsMappedData]]:
        """Collect thresholds of weights and activations"""

        model_layers_params = self.get_model_layers_params()
        activations_thresholds = ThresholdsMappedData({
            layer_name: layer_params.get('activations_thresholds')
            for layer_name, layer_params in model_layers_params.items()
            if 'activations_thresholds' in layer_params
        })
        weights_thresholds = ThresholdsMappedData({
            layer_name: layer_params['weights_thresholds']
            for layer_name, layer_params in model_layers_params.items()
            if 'weights_thresholds' in layer_params
        })

        return activations_thresholds, weights_thresholds
