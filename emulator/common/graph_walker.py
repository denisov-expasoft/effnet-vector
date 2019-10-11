__all__ = ['GraphWalker']

from collections import deque
from copy import deepcopy
from typing import Any
from typing import Dict
from typing import Iterator
from typing import Set
from typing import Tuple

from emulator.common.layers_conf_parameters import LayerConfigurationParameters as Lcp
from emulator.common.misc_utils import normalize_to_list


class GraphWalker:
    """A tool for iteration over layers of the graph specified via the model configuration"""

    def __init__(self, cfg: Dict[str, Any]):
        """"""
        self._all = set(cfg)  # set of all layers names
        self._sources: Set[str] = set()
        self._sinks: Set[str] = set()
        self._forward_mapping: Dict[str, Set[str]] = {}
        self._backward_mapping: Dict[str, Set[str]] = {}

        for name, params in cfg.items():
            inputs = params.get(Lcp.ATTR_COMMON_BOTTOM.value, [])
            inputs = normalize_to_list(inputs)

            # If the current layer doesn't have any inputs,
            # we mark it as a "source" layer
            if not inputs:
                self._sources.add(name)
                continue

            # Update output mapping information of each
            # input of the current layer
            for input_layer_name in inputs:
                mapping = self._forward_mapping.setdefault(input_layer_name, set())
                mapping.add(name)

            # Update inputs info of the current layer
            mapping = self._backward_mapping.setdefault(name, set())
            mapping |= set(inputs)

        # The output layers of the model are those which
        # do not have any output
        self._sinks = self._all - set(self._forward_mapping)
        self._check_graph()

    def copy_sources(self) -> Set[str]:
        """Get input layers of the graph"""
        return self._sources.copy()

    def copy_sinks(self) -> Set[str]:
        """Get output layers of the graph"""
        return self._sinks.copy()

    def copy_forward_mapping(self) -> Dict[str, Set[str]]:
        """Get a dictionary with layers forward mapping"""
        return deepcopy(self._forward_mapping)

    def copy_backward_mapping(self) -> Dict[str, Set[str]]:
        """Get a dictionary with layers backward mapping"""
        return deepcopy(self._backward_mapping)

    def _check_graph(self) -> None:
        number_of_layers = len(self._all)
        visited_layers = set()
        for layer_name, _, _ in self.walk_forward_iter():
            if layer_name in visited_layers:
                raise RuntimeError(f'got circle in graph (layer name = {layer_name})')

            visited_layers.add(layer_name)

        if len(visited_layers) != number_of_layers:
            raise RuntimeError(f'got unused layers in graph ({self._all - visited_layers})')

    def _check_input_layers(self, input_layers: Set[str]) -> None:
        if not isinstance(input_layers, set):
            raise TypeError('Initial layers must be provided via a set of layers names')

        if not input_layers.issubset(self._all):
            raise ValueError(
                'Some of the specified layers are not presented in the graph'
            )

    def walk_forward_iter(self) -> Iterator[Tuple[str, bool, bool]]:
        """Iterate over the graph layers in a forward-propagation manner"""
        return self._create_iter(
            self._sources,
            self._sinks,
            self._forward_mapping,
            self._backward_mapping,
        )

    def walk_backward_iter(self) -> Iterator[Tuple[str, bool, bool]]:
        """Iterate over the graph layers in a backward-propagation manner"""
        return self._create_iter(
            self._sinks,
            self._sources,
            self._backward_mapping,
            self._forward_mapping,
        )

    def walk_forward_from_iter(self, initial_layers: Set[str]) -> Iterator[Tuple[str, bool, bool]]:
        """Iterate over the graph layers in a forward-propagation manner"""
        self._check_input_layers(initial_layers)
        return self._create_iter(
            initial_layers,
            self._sinks,
            self._forward_mapping,
            self._backward_mapping,
        )

    def walk_backward_from_iter(self, initial_layers: Set[str]) -> Iterator[Tuple[str, bool, bool]]:
        """Iterate over the graph layers in a backward-propagation manner"""
        self._check_input_layers(initial_layers)
        return self._create_iter(
            initial_layers,
            self._sources,
            self._backward_mapping,
            self._forward_mapping,
        )

    @staticmethod
    def _create_iter(
            sources: Set[str],
            sinks: Set[str],
            forward_mapping: Dict[str, Set[str]],
            backward_mapping: Dict[str, Set[str]],
    ) -> Iterator[Tuple[str, bool, bool]]:
        visited_layers = set()
        layers_queue = deque(sources)
        while layers_queue:
            current_layer_name = layers_queue.popleft()
            visited_layers.add(current_layer_name)
            is_source = current_layer_name in sources
            is_sink = current_layer_name in sinks

            yield current_layer_name, is_source, is_sink

            layers_queue += GraphWalker._collect_ready_layers(
                current_layer_name,
                visited_layers,
                forward_mapping,
                backward_mapping,
            )

    @staticmethod
    def _is_layer_ready(
            layer_name: str,
            visited_layers: Set[str],
            backward_mapping: Dict[str, Set[str]],
    ) -> bool:
        mapping = backward_mapping.get(layer_name, set())
        return mapping.issubset(visited_layers)

    @staticmethod
    def _collect_ready_layers(
            layer_name: str,
            visited_layers: Set[str],
            forward_mapping: Dict[str, Set[str]],
            backward_mapping: Dict[str, Set[str]],
    ) -> Set[str]:
        return {
            output_name
            for output_name in forward_mapping.get(layer_name, {})
            if GraphWalker._is_layer_ready(output_name, visited_layers, backward_mapping)
        }
