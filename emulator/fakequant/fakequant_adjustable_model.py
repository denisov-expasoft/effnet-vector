__all__ = [
    'AdjustableThresholdsModel',
    'AdjustableWeightsModel',
]

import uuid
from pathlib import Path
from typing import List
from typing import Optional
from typing import Union

import tensorflow as tf

from emulator.common import LayerConfigurationParameters as Lcp
from emulator.common import SupportedLayerTypes as Slt
from emulator.common.data_types import TWeightsCfg
from emulator.fakequant.calibrators import ThresholdsMappedData
from emulator.fakequant.fakequant_adjustable_meta_layers import FAKEQUANT_ADJUSTABLE_THRESHOLDS_METALAYERS
from emulator.fakequant.fakequant_adjustable_meta_layers import FAKEQUANT_ADJUSTABLE_WEIGHTS_METALAYERS
from emulator.fakequant.fakequant_adjustable_meta_layers import AdjWeightsMatrixOpsMetaLayer
from emulator.fakequant.fakequant_meta_layer import FQMetaLayerEnvelope
from emulator.fakequant.fakequant_model import FakeQuantModel
from emulator.layers import BaseGraphLayer


class AdjustableThresholdsModel(FakeQuantModel):

    def __init__(
            self,
            cfg_or_cfg_path: Union[str, Path, dict],
            weights: TWeightsCfg,
            activations_threshold_data: ThresholdsMappedData,
            weights_threshold_data: ThresholdsMappedData,
    ):
        self._variables_checkpoint_path: Optional[Path] = None

        super().__init__(
            cfg_or_cfg_path=cfg_or_cfg_path,
            weights=weights,
            activations_threshold_data=activations_threshold_data,
            weights_threshold_data=weights_threshold_data,
        )

    @property
    def variables_checkpoint_path(self) -> Optional[Path]:
        return self._variables_checkpoint_path

    @variables_checkpoint_path.setter
    def variables_checkpoint_path(self, checkpoint_path: Optional[Union[str, Path]]):
        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)

        self._variables_checkpoint_path = checkpoint_path

    @staticmethod
    def _clip_grad_op(operation, gradient):
        op_input, op_input_min, op_input_max = operation.inputs
        condition = tf.logical_or(
            tf.less(op_input, op_input_min),
            tf.greater(op_input, op_input_max),
        )
        gradient = tf.where(
            condition,
            tf.zeros_like(
                gradient,
                name='zero_grad',
            ),
            gradient,
        )

        min_gradient = tf.constant(0, name='constant_min_gradient')
        max_gradient = tf.constant(0, name='constant_max_gradient')

        return gradient, min_gradient, max_gradient

    def _create_model(self, weights: TWeightsCfg) -> None:
        clip_by_val_gradient_name = f'MyClipGrad_{uuid.uuid4()}'
        tf.RegisterGradient(clip_by_val_gradient_name)(self._clip_grad_op)
        op_type_map = {
            'Round': 'Identity',
            'Floor': 'Identity',
            'ClipByValue': clip_by_val_gradient_name,
        }

        # pylint: disable=not-context-manager
        with self._graph.gradient_override_map(op_type_map):
            super()._create_model(weights)
        # pylint: enable=not-context-manager

    @staticmethod
    def _create_layer(layer_inputs: List[BaseGraphLayer], layer_cfg: dict) -> BaseGraphLayer:
        layer_type = layer_cfg[Lcp.ATTR_COMMON_TYPE.value]
        layer_type = Slt(layer_type)

        layer_class = FAKEQUANT_ADJUSTABLE_THRESHOLDS_METALAYERS.get(layer_type, FQMetaLayerEnvelope)

        return layer_class.create_layer(inputs=layer_inputs, **layer_cfg)

    def _initialize_vars(self) -> None:
        session = tf.get_default_session()
        global_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES
        )
        trainable_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES
        )

        vars_initializer = [var.initializer for var in global_vars]
        if vars_initializer:
            session.run(vars_initializer)

        if self.variables_checkpoint_path is None:
            return

        saver = tf.train.Saver(var_list=trainable_vars)
        saver.restore(session, str(self.variables_checkpoint_path))

    def _create_session(self) -> tf.Session:
        session = super()._create_session()

        # pylint: disable=not-context-manager
        with self._graph.as_default():
            with session.as_default():
                self._initialize_vars()
        # pylint: enable=not-context-manager

        return session


class AdjustableWeightsModel(AdjustableThresholdsModel):
    @staticmethod
    def _create_layer(layer_inputs: List[BaseGraphLayer], layer_cfg: dict) -> BaseGraphLayer:
        layer_type = layer_cfg[Lcp.ATTR_COMMON_TYPE.value]
        layer_type = Slt(layer_type)
        layer_class = FAKEQUANT_ADJUSTABLE_WEIGHTS_METALAYERS.get(layer_type, FQMetaLayerEnvelope)
        return layer_class.create_layer(inputs=layer_inputs, **layer_cfg)

    def get_network_weights(self) -> TWeightsCfg:
        """Collect weights of the network"""

        weights: TWeightsCfg = {}

        with self._create_session():
            for layer_name in self.get_layers_names():
                layer: AdjWeightsMatrixOpsMetaLayer = self._layers[layer_name]

                if isinstance(layer, AdjWeightsMatrixOpsMetaLayer):
                    weights[layer_name] = layer.get_weights()

        return weights
