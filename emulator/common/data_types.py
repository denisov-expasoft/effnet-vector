__all__ = [
    'TBias',
    'TWeights',
    'TArrayOrTensor',

    'TSingleInputDataset',
    'TMultipleInputDataset',
    'TDataset',
    'TLayerWeightsCfg',
    'TWeightsCfg',
    'TCalibrationDataset',

    'TMinMaxThresholdsData',

    'TPyTransform',
]

from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import tensorflow as tf

TArrayOrTensor = Union[np.ndarray, tf.Tensor]
TWeights = TArrayOrTensor
TBias = TArrayOrTensor

TSingleInputDataset = List[np.ndarray]
TMultipleInputDataset = Dict[str, TSingleInputDataset]
TDataset = Union[TSingleInputDataset, TMultipleInputDataset]

TLayerWeightsCfg = Dict[str, np.ndarray]
TWeightsCfg = Dict[str, TLayerWeightsCfg]

TCalibrationDataset = TDataset

TMinMaxThresholdsData = Union[Tuple[np.ndarray, np.ndarray], Tuple[tf.Tensor, tf.Tensor]]

TPyTransform = Callable[[np.ndarray], np.ndarray]
