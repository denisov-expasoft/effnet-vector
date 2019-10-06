# Perform model evaluation

import pickle
from pathlib import Path
from typing import Tuple
from typing import Union

import emulator
from emulator.common.data_utils import load_weights
from emulator.dataset import batch_stream_from_records
from emulator.dataset import imagenet_evaluate_tflite
from emulator.fakequant.calibrators import ThresholdsMappedData


def _load_thresholds(path: Union[str, Path]) -> Tuple[ThresholdsMappedData, ThresholdsMappedData]:
    path = Path(path)
    with path.open('rb') as file:
        data = pickle.load(file)
    return data


def main():
    emulator.turn_logging_off()

    a_ths, w_ths = _load_thresholds('model-data/user_trained_thresholds.pickle')
    weights = load_weights('model-data/user_trained_weights.pickle')

    new_image_size = 256

    fakequant_model = emulator.TFliteFakeQuantModel(
        'model-data/fakequant.json',
        weights,
        a_ths,
        w_ths,
        [None, new_image_size, new_image_size, 3],
    )

    fakequant_model.export_graph('model-data/user_trained_graph_with_fakequant.pb')
    fakequant_model.export_tflite_model('model-data/user_trained_model.tflite')

    # Evaluation
    tflite_batch_stream = batch_stream_from_records(
        Path('dataset-data/val_set'),
        batch_size=1,
        output_image_size=new_image_size,
        preprocess_fun=lambda image: image,  # image already has values from 0 to 255
        crop_fraction=1,
    )

    emulator.turn_logging_on()
    imagenet_evaluate_tflite('model-data/user_trained_model.tflite', tflite_batch_stream, log_every_n_batches=100)


if __name__ == '__main__':
    main()
