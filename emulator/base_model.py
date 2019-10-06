__all__ = [
    'BaseModel'
]

import gc
import logging
from abc import ABCMeta
from abc import abstractmethod
from copy import deepcopy
from itertools import filterfalse
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List
from typing import Set
from typing import Tuple
from typing import Union

import numpy as np
import tensorflow as tf

from emulator.common import DatasetError
from emulator.common import GraphError
from emulator.common import GraphWalker
from emulator.common import LayerConfigurationParameters as Lcp
from emulator.common import exceptions
from emulator.common import normalize_to_list
from emulator.common.cfg_utils import load_cfg
from emulator.common.data_types import TDataset
from emulator.common.data_types import TLayerWeightsCfg
from emulator.common.data_types import TMultipleInputDataset
from emulator.common.data_types import TWeightsCfg
from emulator.layers import BaseGraphLayer

_LOGGER = logging.getLogger('emulator.model')


class BaseModel(metaclass=ABCMeta):

    def __init__(
            self,
            cfg_or_cfg_path: Union[str, Path, dict],
            weights: TWeightsCfg,
    ):
        self._cfg = load_cfg(cfg_or_cfg_path)
        self._check_configuration()

        self._graph = tf.Graph()
        self._input_layers_names = set()
        self._output_layers_names = set()
        self._layers: Dict[str, BaseGraphLayer] = dict()
        self._graph_walker = GraphWalker(self._cfg)
        self._input_anchors: Dict[str, str] = dict()
        self._output_anchors: Dict[str, str] = dict()
        self._weights = deepcopy(weights)

        _LOGGER.info('Start creating graph')

        # pylint: disable=not-context-manager
        with self._graph.as_default():
            self._create_model(self._weights)
        # pylint: enable=not-context-manager

        _LOGGER.info('End of graph assembling')

    @abstractmethod
    def _check_configuration(self) -> None:
        pass

    @property
    def graph(self) -> tf.Graph:
        """Returns instance of the TensorFlow computational graph"""
        return self._graph

    @property
    def graph_inputs(self) -> List[tf.Tensor]:
        return [
            self._layers[layer_name].backend_output
            for layer_name in self._input_layers_names
        ]

    @property
    def graph_outputs(self) -> List[tf.Tensor]:
        return [
            self._layers[layer_name].backend_output
            for layer_name in self._output_layers_names
        ]

    @property
    def graph_info(self) -> Tuple[tf.Graph, List[tf.Tensor], List[tf.Tensor]]:
        """Returns instance of the TensorFlow computational graph
        and its input/output tensors.
        """
        graph_inputs = self.graph_inputs
        graph_outputs = self.graph_outputs

        return self._graph, graph_inputs, graph_outputs

    @property
    def cfg(self) -> dict:
        """Returns the configuration used for building
        the current instance of BaseModel.
        """
        return deepcopy(self._cfg)

    def get_layers_names(self, exclude_input: bool = False) -> List[str]:
        """Returns the list of all layers names.

        Optionally, names of input layers can be excluded from the list.
        """
        layers_names = self._layers.keys()
        if exclude_input:
            layers_names = filterfalse(
                lambda name: name in self._input_layers_names,
                layers_names,
            )

        return list(layers_names)

    def get_layer_output_shape(self, layer_name) -> Tuple[int, ...]:
        return self._layers[layer_name].output_shape

    def export_graph(self, target_path: Union[str, Path]) -> None:
        """Saves the model using the ProtocolBuffer format.

        The resulting GraphDef object is stored as binary data.
        """
        target_path = Path(target_path)
        result = tf.train.write_graph(
            self._graph,
            logdir=str(target_path.parent),
            name=target_path.name,
            as_text=False,
        )

        _LOGGER.info(f'graph saved as {result}')

    def export_to_tensorboard(
            self, target_dir: Path,
            filename_suffix: str = '_custom_graph',
    ) -> None:
        """Saves the TensorFlow event for furthur visualization in TensorBoard"""
        tf.summary.FileWriter(
            graph=self._graph,
            logdir=str(target_dir),
            filename_suffix=filename_suffix
        )

    def get_model_layers_params(self, layers_names: Union[str, List[str]] = None) -> dict:
        """Get static data of the BaseModel's layers.

        It collects the basic information about layers such as the quantization parameters
        (scales, shifts), look-up tables, some operation defining constants, etc.

        The output is a dictionary in which each key corresponds to a layer name of
        the assembled model.

        Parameters
        ----------
        layers_names : None, str, list
            Name(s) of layers, which input-independent parameters should be returned.

            - If `layers` is `None`, the method will return parameters of all layers.
            - If 'layers' is string or list of strings, the method will return the parameters
              of specified layers only

        Returns
        -------
        dict
            Parameters of the model's layers in the following format::

                {
                    'layer_name': {
                        'parameter_1': val_1,
                        'parameter_2': val_2,
                        ...
                    },
                    ...
                }

        Raises
        ------
        GraphError
            If layer(s) with the specified name(s) cannot be found in the graph
        """
        layers_names = self._check_and_normalize_layers_names(layers_names)

        result = {}
        with self._create_session():
            for layer_name in layers_names:
                layer = self._layers[layer_name]
                layer_params = layer.get_meaningful_properties()
                result[layer_name] = layer_params

        return result

    def get_output_data(
            self,
            dataset: TDataset,
            layers_names: Union[str, List[str]] = None,
            fold_batches_result: bool = False
    ) -> Dict[str, np.ndarray]:
        layers_names = self._check_and_normalize_layers_names(layers_names)
        dataset = self._check_and_normalize_dataset(dataset)

        backend_tensors = {
            layers_name: self._layers[layers_name].backend_output
            for layers_name in layers_names
        }
        result = self._get_output_data(backend_tensors, dataset, fold_batches_result)

        return result

    def walk_by_output_data(
            self,
            dataset: TDataset,
            layer_output_data_cb: Callable[[str, np.ndarray], None],
            layers_names: Union[str, List[str]] = None,
    ) -> None:
        layers_names = self._check_and_normalize_layers_names(layers_names)
        dataset = self._check_and_normalize_dataset(dataset)

        layers = {
            layer_name: self._layers[layer_name]
            for layer_name in layers_names
        }
        dataset_feeds = self._split_dataset_by_feeds(dataset)

        with self._create_session() as session:
            all_fetches = [
                layer.backend_output
                for name, layer in layers.items()
                if name not in self._input_layers_names
            ]

            for i, batch_feed in enumerate(dataset_feeds):
                partial_run_handler = session.partial_run_setup(
                    fetches=all_fetches,
                    feeds=list(batch_feed.keys()),
                )

                for layer_name, layer in layers.items():
                    if layer_name in self._input_layers_names:
                        layer_data = deepcopy(dataset[layer_name][i])
                        layer_output_data_cb(layer_name, layer_data)
                        continue

                    layer_data = session.partial_run(
                        partial_run_handler,
                        fetches=layer.backend_output,
                        feed_dict=batch_feed,
                    )
                    batch_feed = None
                    layer_output_data_cb(layer_name, layer_data)
        # we should cleanup garbage manually
        gc.collect()

    def create_session(self) -> tf.Session:
        return self._create_session()

    def _create_session(self) -> tf.Session:
        return tf.Session(graph=self._graph)

    def _create_model(self, weights: TWeightsCfg) -> None:
        graph_walker = self._graph_walker

        input_layers_names = graph_walker.copy_sources()
        output_layers_names = graph_walker.copy_sinks()

        for layer_name, _, _ in graph_walker.walk_forward_iter():

            layer_weights = weights.pop(layer_name, {})
            layer_inputs, layer_cfg = self._prepare_layer_inputs_and_cfg(layer_name, layer_weights)
            layer_type = layer_cfg[Lcp.ATTR_COMMON_TYPE.value]

            _LOGGER.info(
                f'Create "{layer_name}" layer with type "{layer_type}"'
            )

            try:
                with tf.name_scope(layer_name + '/'):
                    layer = self._create_layer(layer_inputs, layer_cfg)
                    self._layers[layer_name] = layer
            except Exception as err:
                raise exceptions.ModelBuildingError(
                    f'Unable to build "{layer_name}"'
                ) from err

            output_layer_shape = layer.backend_output.get_shape()
            _LOGGER.debug(f'Output layer shape: "{output_layer_shape.as_list()}"')

        self._register_input_and_output_layers(input_layers_names, output_layers_names)

    def _register_input_and_output_layers(
            self,
            input_layers_names: Set[str],
            output_layers_names: Set[str],
    ) -> None:
        _LOGGER.info('Register input and output layers of the model')

        def get_anchor(layer_name: str) -> str:
            layer_cfg = self._cfg[layer_name]
            return layer_cfg[Lcp.ATTR_IO_GRAPH_ANCHOR.value]

        self._input_anchors = {
            get_anchor(layer_name): layer_name
            for layer_name in input_layers_names
        }

        self._output_anchors = {
            get_anchor(layer_name): layer_name
            for layer_name in output_layers_names
        }

        self._input_layers_names = input_layers_names.copy()
        self._output_layers_names = output_layers_names.copy()

    def _prepare_layer_inputs_and_cfg(
            self,
            layer_name: str,
            layer_weights: TLayerWeightsCfg,
    ) -> Tuple[List[BaseGraphLayer], dict]:
        # Create a copy of the configuration of the specified layer,
        # so we can add and remove fields from it without affecting
        # the initial configuration data
        layer_cfg = deepcopy(self._cfg[layer_name])
        layer_cfg.update(layer_weights)
        # Remove weights shape from the configuration block
        layer_cfg.pop(Lcp.ATTR_WEIGHTS_SHAPE.value, None)
        # Layer input
        inputs_names = layer_cfg.pop(Lcp.ATTR_COMMON_BOTTOM.value, None)
        inputs_names = normalize_to_list(inputs_names)
        layer_inputs = [self._layers[name] for name in inputs_names]

        return layer_inputs, layer_cfg

    @staticmethod
    @abstractmethod
    def _create_layer(layer_inputs: List[BaseGraphLayer], layer_cfg: dict) -> BaseGraphLayer:
        pass

    def _check_and_normalize_layers_names(
            self,
            layers_names: Union[str, List[str]] = None,
    ) -> List[str]:
        if layers_names is None:
            layers_names = self.get_layers_names()

        layers_names = normalize_to_list(layers_names)

        unknown_layers = set(layers_names) - set(self._layers.keys())
        if unknown_layers:
            raise GraphError(f'Layers "{unknown_layers}" are not found')

        return layers_names

    @staticmethod
    def _get_number_of_batches(dataset) -> int:
        number_of_batches = {len(input_data) for input_data in dataset.values()}
        if len(number_of_batches) != 1:
            raise DatasetError(
                'dataset must contain equal number of batches for every input'
            )

        return number_of_batches.pop()

    def _check_and_normalize_dataset(self, dataset: TDataset) -> TMultipleInputDataset:
        if isinstance(dataset, list):
            if len(self._input_layers_names) != 1:
                raise DatasetError(
                    'got single input dataset for multi input model'
                )

            [input_name] = self._input_layers_names
            return {
                input_name: dataset
            }

        self._get_number_of_batches(dataset)

        return dataset

    def _split_dataset_by_feeds(self, dataset):
        number_of_batches = self._get_number_of_batches(dataset)
        dataset_by_batches = [
            {
                self._layers[name].backend_output: data[i]
                for name, data in dataset.items()
            }
            for i in range(number_of_batches)
        ]

        return dataset_by_batches

    def _get_output_data(
            self,
            backend_tensors: Dict[str, tf.Tensor],
            dataset: TMultipleInputDataset,
            fold_batches_result: bool = False
    ) -> Union[Dict[str, TDataset], Dict[str, np.ndarray]]:
        result = {}
        backend_output_tensors = {}

        for layer_name, layer_backend_output in backend_tensors.items():
            if layer_name in self._input_layers_names:
                result[layer_name] = deepcopy(dataset[layer_name])
            else:
                result[layer_name] = []
                backend_output_tensors[layer_name] = layer_backend_output

        with self._create_session() as session:
            dataset_feeds = self._split_dataset_by_feeds(dataset)
            for batch_feed in dataset_feeds:
                one_batch_result = session.run(
                    fetches=backend_output_tensors,
                    feed_dict=batch_feed,
                )

                for layer_name, layer_one_batch_result in one_batch_result.items():
                    result[layer_name].append(layer_one_batch_result)

        # we should cleanup garbage manually
        gc.collect()

        if fold_batches_result:
            for layer_name, one_layer_result in result.items():
                result[layer_name] = np.concatenate(one_layer_result, axis=0)

        return result
