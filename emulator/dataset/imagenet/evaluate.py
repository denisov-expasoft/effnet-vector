__all__ = [
    'imagenet_evaluate',
]

import logging
from time import time
from typing import Tuple

from emulator.base_model import BaseModel
from emulator.dataset.base.batch_stream import BaseBatchStream
from emulator.dataset.imagenet.metric_accumulator import ImagenetMetricAccumulator

_LOGGER = logging.getLogger('emulator.dataset.evaluate')


def imagenet_evaluate(
        model: BaseModel,
        batch_stream: BaseBatchStream, *,
        log_every_n_batches=10,
) -> Tuple[float, float]:

    [graph_input] = model.graph_inputs
    [graph_output] = model.graph_outputs

    number_of_classes = graph_output.shape
    number_of_classes = int(number_of_classes[-1])
    labels_offset = number_of_classes % 1000
    metric_accumulator = ImagenetMetricAccumulator(labels_offset)

    dt, prev_dt = 0, 0

    with model.create_session() as session:
        t_0 = time()

        for i, (batch_data, labels) in enumerate(batch_stream, start=1):
            predictions = session.run(graph_output, {graph_input: batch_data})

            t_1 = time()
            metric_accumulator.on_batch_result(predictions, labels)
            dt += t_1 - t_0

            if i % log_every_n_batches == 0:
                metric_as_str = metric_accumulator.as_human_readable_str()
                _LOGGER.info(
                    f'Batch #{i}: {metric_as_str}'
                )
                _LOGGER.info(
                    f'Batch #{i}: prediction_dt = {dt - prev_dt:.3f}s'
                )
                prev_dt = dt

            t_0 = time()

    metric_as_str = metric_accumulator.as_human_readable_str()
    _LOGGER.info(
        f'Total: {metric_as_str}'
    )
    _LOGGER.info(
        f'Total: prediction_dt = {dt:.3f}s'
    )

    return metric_accumulator.get_current_metric()
