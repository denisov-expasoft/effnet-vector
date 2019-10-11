__all__ = [
    'TThresholdsMapping',
    'TMinMaxThresholdsData',
    'TThresholdsData',
    'TLayerUnionMap',
    'ThresholdsMappedData',
    'load_thresholds_mapped_data',
    'save_thresholds_mapped_data',
]

import pickle
from pathlib import Path
from typing import Dict
from typing import Tuple
from typing import Union

import numpy as np
import tensorflow as tf

TThresholdsMapping = Dict[str, Tuple[np.ndarray, np.ndarray]]
TMinMaxThresholdsData = Union[Tuple[np.ndarray, np.ndarray], Tuple[tf.Tensor, tf.Tensor]]
TThresholdsData = Union[np.ndarray, tf.Tensor]
TLayerUnionMap = Dict[str, str]


class ThresholdsMappedData:
    def __init__(
            self,
            thresholds_value_map: TThresholdsMapping,
    ):
        if thresholds_value_map is None:
            raise ValueError('\'thresholds_value_map\' cannot be None')

        self._union_threshold_value_map = thresholds_value_map

    @property
    def thresholds_dict(self) -> TThresholdsMapping:
        return self._union_threshold_value_map


def save_thresholds_mapped_data(thresholds_mapped_data: ThresholdsMappedData, path: Path) -> None:
    if isinstance(thresholds_mapped_data, ThresholdsMappedData):
        path = Path(path)
        with path.open('wb') as file:
            pickle.dump(thresholds_mapped_data, file)
    else:
        raise ValueError(
            f'\'thresholds_mapped_data\' positional variable '
            f'has wrong type: "{type(thresholds_mapped_data)}"'
        )


def load_thresholds_mapped_data(path: Path) -> ThresholdsMappedData:
    path = Path(path)

    with path.open('rb') as file:
        probably_threshold_mapped_data = pickle.load(file)

    if not isinstance(probably_threshold_mapped_data, ThresholdsMappedData):
        raise ValueError(f'Pickled object has wrong type: "{type(probably_threshold_mapped_data)}"')

    return probably_threshold_mapped_data
