__all__ = [
    'BaseDatasetTransformation',
]

import multiprocessing
from abc import ABCMeta
from abc import abstractmethod
from typing import Optional

import numpy as np
import tensorflow as tf

from emulator.common.data_types import TPyTransform


class BaseDatasetTransformation(metaclass=ABCMeta):
    def __init__(
            self, *,
            preprocess_fun: TPyTransform = None,
            num_parallel_calls: int = None,
    ):
        self._preprocess_fun = self._safe_preprocess_fun(preprocess_fun)

        if num_parallel_calls is None:
            num_parallel_calls = multiprocessing.cpu_count()

        self._num_parallel_calls = num_parallel_calls

    @staticmethod
    def _safe_preprocess_fun(preprocess_fun: Optional[TPyTransform]) -> Optional[TPyTransform]:

        def preprocess_fun_wrapper(data: np.ndarray) -> np.ndarray:
            result = preprocess_fun(data)
            return result.astype(np.float32)

        return preprocess_fun_wrapper

    @abstractmethod
    def _prepare_data(self, data: tf.Tensor) -> tf.Tensor:
        if self._preprocess_fun:
            return tf.py_func(self._preprocess_fun, [data], tf.float32)

        return data

    def __call__(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.map(
            self._prepare_data,
            num_parallel_calls=self._num_parallel_calls,
        )
