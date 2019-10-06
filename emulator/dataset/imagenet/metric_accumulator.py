__all__ = [
    'ImagenetMetricAccumulator',
]

from typing import Tuple

import numpy as np


class ImagenetMetricAccumulator:
    def __init__(self, labels_offset: int = 0):
        self._labels_offset = labels_offset
        self._top_1 = 0
        self._top_5 = 0
        self._count = 0

    def get_current_metric(self) -> Tuple[float, float]:
        top_1 = self._top_1 / self._count if self._count else 0
        top_5 = self._top_5 / self._count if self._count else 0

        return top_1, top_5

    def as_human_readable_str(self) -> str:
        top_1, top_5 = self.get_current_metric()
        return f'top 1 = {top_1:.4f}; top 5 = {top_5:.4f}'

    def on_batch_result(self, predicted_result: np.ndarray, true_result: np.ndarray) -> None:
        true_result = true_result + self._labels_offset

        ndim = np.ndim(predicted_result)
        if ndim > 2:
            squeeze_axis = tuple(range(1, ndim - 1))
            predicted_result = np.squeeze(predicted_result, axis=squeeze_axis)

        true_result = np.expand_dims(true_result, -1)

        batch_top_5_labels = np.argsort(predicted_result, axis=-1)
        batch_top_5_labels = batch_top_5_labels[..., ::-1]
        batch_top_5_labels = batch_top_5_labels[..., 0:5]

        batch_top_5 = batch_top_5_labels == true_result
        batch_top_1 = batch_top_5[..., 0]
        batch_top_5 = np.sum(batch_top_5)
        batch_top_1 = np.sum(batch_top_1)

        self._top_1 += batch_top_1
        self._top_5 += batch_top_5
        self._count += len(predicted_result)
