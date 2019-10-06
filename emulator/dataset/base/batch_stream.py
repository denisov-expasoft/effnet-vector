__all__ = [
    'BaseBatchStream',
    'RecordsWithLabelsBatchStream',
]

from abc import ABCMeta
from abc import abstractmethod
from pathlib import Path
from typing import Callable
from typing import List
from typing import Union

import tensorflow as tf

from emulator.common import normalize_to_list


class BaseBatchStream(metaclass=ABCMeta):

    def __init__(
            self, batch_size: int, *,
            prefetch_size: int = None,
    ):
        self._batch_size = batch_size
        if prefetch_size is None:
            self._prefetch_size = -1

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def prefetch_size(self) -> int:
        return self._prefetch_size

    def __iter__(self):
        config = tf.ConfigProto(device_count={'GPU': 0})
        graph = tf.Graph()
        dataset = self.create_dataset(graph)
        with tf.Session(graph=graph, config=config) as session:
            dataset_iter = dataset.make_one_shot_iterator()
            next_element = dataset_iter.get_next()
            while True:
                try:
                    yield session.run(next_element)
                except tf.errors.OutOfRangeError:
                    break

    @abstractmethod
    def _create_dataset(self) -> tf.data.Dataset:
        pass

    def create_dataset(self, graph: tf.Graph) -> tf.data.Dataset:
        with graph.as_default():
            dataset = self._create_dataset()
            dataset = dataset.batch(self._batch_size)
            dataset = dataset.prefetch(self._prefetch_size)

            return dataset


class RecordsWithLabelsBatchStream(BaseBatchStream):

    def __init__(
            self,
            records_paths: Union[Path, List[Path]],
            data_feature: str,
            label_feature: str,
            batch_size: int, *,
            transform_dataset_fun: Callable[[tf.data.Dataset], tf.data.Dataset] = None,
            prefetch_size: int = None,
    ):
        if not records_paths:
            raise ValueError('List of records cannot be empty')

        self._records_paths = normalize_to_list(records_paths)
        self._data_feature = data_feature
        self._label_feature = label_feature
        self._transform_dataset_fun = transform_dataset_fun

        super().__init__(batch_size, prefetch_size=prefetch_size)

    def _parse_record(self, data_as_string):
        features = tf.parse_single_example(
            data_as_string,
            features={
                self._data_feature: tf.FixedLenFeature([], tf.string),
                self._label_feature: tf.FixedLenFeature([], tf.int64),
            },
        )
        data = features[self._data_feature]
        label = features[self._label_feature]

        return data, label

    def _create_dataset(self) -> tf.data.Dataset:
        records_paths = [str(path) for path in self._records_paths]
        dataset = tf.data.TFRecordDataset(records_paths)
        dataset = dataset.map(self._parse_record)

        dataset_data = dataset.map(lambda data, _: data)
        dataset_labels = dataset.map(lambda _, label: label)

        if self._transform_dataset_fun:
            dataset_data = self._transform_dataset_fun(dataset_data)

        dataset = tf.data.Dataset.zip(
            (dataset_data, dataset_labels)
        )

        return dataset
