# Perform model evaluation for a competition entry

from pathlib import Path

import emulator
from emulator.dataset import batch_stream_from_records
from emulator.dataset import imagenet_evaluate_tflite


def main():

    # Evaluation
    tflite_batch_stream = batch_stream_from_records(
        Path('dataset-data/val_set'),
        batch_size=1,
        output_image_size=256,
        preprocess_fun=lambda image: image,  # image already has values from 0 to 255
        crop_fraction=1,
    )

    emulator.turn_logging_on()
    imagenet_evaluate_tflite('model-data/ready_to_use_model.tflite', tflite_batch_stream, log_every_n_batches=100)


if __name__ == '__main__':
    main()
