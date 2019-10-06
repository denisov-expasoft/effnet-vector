__all__ = [
    'imagenet_evaluate',
    'imagenet_evaluate_tflite',
]

import logging
from pathlib import Path
from time import time
from typing import Tuple
from typing import Union

import numpy as np
from tensorflow.contrib import lite

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


def imagenet_evaluate_tflite(
        tflite_model_path: Union[str, Path],
        batch_stream: BaseBatchStream, *,
        log_every_n_batches=500,
) -> Tuple[float, float]:
    tflite_model_path = Path(tflite_model_path)
    interpreter = lite.Interpreter(model_path=tflite_model_path.as_posix())
    interpreter.allocate_tensors()
    input_tensor_index = interpreter.get_input_details()[0]['index']
    output_tensor_index = interpreter.get_output_details()[0]['index']

    number_of_classes = interpreter.get_output_details()[0]['shape']
    number_of_classes = int(number_of_classes[-1])
    labels_offset = number_of_classes % 1000
    metric_accumulator = ImagenetMetricAccumulator(labels_offset)

    dt, prev_dt = 0, 0

    t_0 = time()

    for i, (batch_data, labels) in enumerate(batch_stream, start=1):

        interpreter.set_tensor(input_tensor_index, batch_data.astype(np.uint8))
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_tensor_index)

        t_1 = time()
        metric_accumulator.on_batch_result(predictions, labels)
        dt += t_1 - t_0

        if (i + 1) % log_every_n_batches == 0:
            metric_as_str = metric_accumulator.as_human_readable_str()
            _LOGGER.info(
                f'Batch #{i + 1}: {metric_as_str}'
            )
            _LOGGER.info(
                f'Batch #{i + 1}: prediction_dt = {dt - prev_dt:.3f}s'
            )
            prev_dt = dt

        t_0 = time()

    # Final metric logging
    metric_as_str = metric_accumulator.as_human_readable_str()
    _LOGGER.info(
        f'Total: {metric_as_str}'
    )
    _LOGGER.info(
        f'Total: prediction_dt = {dt:.3f}s'
    )

    return metric_accumulator.get_current_metric()
