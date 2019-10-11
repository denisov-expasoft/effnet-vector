from abc import ABCMeta
from abc import abstractmethod
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import tensorflow as tf

from emulator.common import Registry

TAxis = Optional[Union[int, Tuple[int, ...]]]

WEIGHTS_CALIBRATORS_REGISTRY = Registry()


class BaseCalibrator(metaclass=ABCMeta):

    @staticmethod
    @abstractmethod
    def to_output_format_np(thresholds_or_scale: np.ndarray):
        pass

    @staticmethod
    @abstractmethod
    def to_output_format_tf(thresholds_or_scale: tf.Tensor):
        pass

    @abstractmethod
    def _get_axis(self, data_shape: Tuple[int, ...]) -> TAxis:
        pass

    def __call__(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        axis = self._get_axis(data.shape)

        min_thresholds = np.min(data, axis=axis, keepdims=True)
        min_thresholds = np.float32(min_thresholds)

        max_thresholds = np.max(data, axis=axis, keepdims=True)
        max_thresholds = np.float32(max_thresholds)

        return min_thresholds, max_thresholds


class MinMaxCalibrator(BaseCalibrator):

    @staticmethod
    def to_output_format_np(thresholds_or_scale: np.ndarray) -> np.ndarray:
        return np.squeeze(thresholds_or_scale.copy())

    @staticmethod
    def to_output_format_tf(thresholds_or_scale: tf.Tensor) -> tf.Tensor:
        return tf.squeeze(thresholds_or_scale)

    def _get_axis(self, _) -> TAxis:
        return None


class ChannelMinMaxCalibrator(MinMaxCalibrator):

    def _get_axis(self, data_shape: Tuple[int, ...]) -> TAxis:
        return tuple(
            range(len(data_shape) - 1)
        )
