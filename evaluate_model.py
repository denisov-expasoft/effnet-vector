# Perform model evaluation

from pathlib import Path

import click

import emulator
from emulator.dataset import batch_stream_from_records
from emulator.dataset import efficientnet_preprocess_function
from emulator.dataset import imagenet_evaluate
from prepare_model_data_for_evaluation import get_cfg_weights_and_quant_data

_IMG_SIZE = 224


@click.command()
@click.option('--img-size', help='The original model', default=_IMG_SIZE, type=int)
def main(img_size):

    cfg, weights, quantization_data = get_cfg_weights_and_quant_data(
        'model-data/regular.json',
        'model-data/fakequant.json',
        'model-data/user_trained_weights.pickle',
        'model-data/user_trained_thresholds_vector.pickle',
        img_size,
    )

    # Build integer model
    integer_model = emulator.TFLiteGpuInterpreter(
        cfg,
        weights,
        quantization_data=quantization_data,
    )

    validation_bach_stream = batch_stream_from_records(
        Path('dataset-data/val_set'),
        batch_size=100,
        output_image_size=img_size,
        preprocess_fun=efficientnet_preprocess_function,
        crop_fraction=1.
    )

    emulator.turn_logging_on()

    imagenet_evaluate(integer_model, validation_bach_stream, log_every_n_batches=50)


if __name__ == '__main__':
    main()
