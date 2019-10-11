# Perform model quantization and training the quantization parameters

import logging
import pickle
from pathlib import Path
from typing import Tuple
from typing import Union

import emulator
from emulator.common.data_types import TWeightsCfg
from emulator.common.data_utils import load_weights
from emulator.dataset import batch_stream_from_records
from emulator.fakequant.calibrators import ThresholdsMappedData
from emulator.dataset import efficientnet_preprocess_function

_LOGGER = logging.getLogger('emulator.quantization')

_IMAGENET_TRAIN_RECORDS_PATHS = [
    Path('dataset-data/train_set0000'),
    Path('dataset-data/train_set0001'),
]
_IMAGENET_VALIDATION_RECORD_PATHS = Path('dataset-data/train_set0000')
_CROP_FRACTION = 1.0
_NUMBER_OF_EPOCH_FOR_THRESHOLDS = 2
_MODEL_DIR = Path('model-data')
_TRAIN_BATCH_SIZE = 32
_IMAGE_OUTPUT_SIZE = 224
_QUANT_CONFIG_PATH = Path('model-data/fakequant.json')
_REGULAR_CONFIG_PATH = Path('model-data/regular.json')
_WEIGHTS_PATH = Path('model-data/weights.pickle')


def _load_thresholds(path: Union[str, Path]) -> Tuple[ThresholdsMappedData, ThresholdsMappedData]:
    path = Path(path)
    with path.open('rb') as file:
        aths , wths = pickle.load(file)

    if not isinstance(aths, ThresholdsMappedData):
        aths = ThresholdsMappedData(aths)

    if not isinstance(wths, ThresholdsMappedData):
        wths = ThresholdsMappedData(wths)

    return aths, wths


def _save_data(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


def _train_thresholds(
        quant_config_path: Path,
        regular_config_path: Path,
        weights: TWeightsCfg,
        initial_a_ths_data: ThresholdsMappedData,
        initial_w_ths_data: ThresholdsMappedData,
) -> Tuple[ThresholdsMappedData, ThresholdsMappedData]:
    emulator.turn_logging_on()
    _LOGGER.info('Prepare the model for quantization thresholds training ...')

    learning_strategy = emulator.fakequant.LearningStrategy(
        optimizer_type='adam',
        initial_lr=3e-4,
        lr_decay=8e-5,
        lr_cos_steps=500,
        lr_cos_phase=3.141592654 * 0.4,
        metric_type='rmse',
    )

    emulator.turn_logging_off()
    thresholds_trainable_model = emulator.AdjustableThresholdsModel(
        cfg_or_cfg_path=quant_config_path,
        weights=weights,
        activations_threshold_data=initial_a_ths_data,
        weights_threshold_data=initial_w_ths_data,
    )
    regular_model = emulator.RegularModel(
        cfg_or_cfg_path=regular_config_path,
        weights=weights,
    )
    thresholds_train_batch_stream = batch_stream_from_records(
        records_paths=_IMAGENET_TRAIN_RECORDS_PATHS,
        batch_size=_TRAIN_BATCH_SIZE,
        output_image_size=_IMAGE_OUTPUT_SIZE,
        crop_fraction=_CROP_FRACTION,
        preprocess_fun=efficientnet_preprocess_function,
    )
    emulator.turn_logging_on()

    best_ckpt_path = emulator.fakequant.train_adjustable_model(
        adjustable_network=thresholds_trainable_model,
        reference_network=regular_model,
        batch_stream=thresholds_train_batch_stream,
        tensorboard_log_dir=_MODEL_DIR / 'trained_thresholds',
        number_of_epochs=_NUMBER_OF_EPOCH_FOR_THRESHOLDS,
        learning_strategy=learning_strategy,
    )

    _LOGGER.info('Checkpoint where the loss-function had its absolute minimum:')
    _LOGGER.info(best_ckpt_path)

    emulator.turn_logging_off()

    return thresholds_trainable_model.get_network_thresholds()


def _train_weights(
        quant_config_path: Path,
        regular_config_path: Path,
        initial_weights: TWeightsCfg,
        a_ths_data: ThresholdsMappedData,
        w_ths_data: ThresholdsMappedData,
):
    emulator.turn_logging_on()
    _LOGGER.info('Prepare the model for quantization-aware weights training ...')

    learning_strategy=emulator.fakequant.LearningStrategy(
        optimizer_type='adam',
        initial_lr=1e-4,
        lr_decay=5e-5,
        lr_cos_steps=1000,
        lr_cos_phase=3.141592654 / 3,
        metric_type='rmse',
    )

    emulator.turn_logging_off()
    weights_trainable_model = emulator.AdjustableWeightsModel(
        cfg_or_cfg_path=quant_config_path,
        weights=initial_weights,
        activations_threshold_data=a_ths_data,
        weights_threshold_data=w_ths_data,
    )
    regular_model = emulator.RegularModel(
        cfg_or_cfg_path=regular_config_path,
        weights=initial_weights,
    )
    weights_train_batch_stream = batch_stream_from_records(
        records_paths=_IMAGENET_TRAIN_RECORDS_PATHS,
        batch_size=_TRAIN_BATCH_SIZE,
        output_image_size=_IMAGE_OUTPUT_SIZE,
        crop_fraction=_CROP_FRACTION,
        preprocess_fun=efficientnet_preprocess_function,
    )
    emulator.turn_logging_on()
    best_ckpt_path = emulator.fakequant.train_adjustable_model(
        adjustable_network=weights_trainable_model,
        reference_network=regular_model,
        batch_stream=weights_train_batch_stream,
        tensorboard_log_dir=_MODEL_DIR / 'trained_weights',
        number_of_epochs=1,
        learning_strategy=learning_strategy,
    )

    _LOGGER.info('Checkpoint where the loss-function had its absolute minimum:')
    _LOGGER.info(best_ckpt_path)

    emulator.turn_logging_off()

    return weights_trainable_model.get_network_weights()


def main():
    emulator.turn_logging_off()

    initial_activations_thresholds, initial_weights_thresholds = _load_thresholds(
        'model-data/initial_thresholds_vector.pickle'
    )

    # Train quantization thresholds
    trained_a_thresholds, trained_w_thresholds = _train_thresholds(
        quant_config_path=_QUANT_CONFIG_PATH,
        regular_config_path=_REGULAR_CONFIG_PATH,
        weights=load_weights(_WEIGHTS_PATH),
        initial_a_ths_data=initial_activations_thresholds,
        initial_w_ths_data=initial_weights_thresholds,
    )
    _save_data((trained_a_thresholds, trained_w_thresholds), 'model-data/user_trained_thresholds_vector.pickle')

    # Quantization-aware weights training
    trained_weights = _train_weights(
        quant_config_path=_QUANT_CONFIG_PATH,
        regular_config_path=_REGULAR_CONFIG_PATH,
        initial_weights=load_weights(_WEIGHTS_PATH),
        a_ths_data=trained_a_thresholds,
        w_ths_data=trained_w_thresholds,
    )
    _save_data(trained_weights, 'model-data/user_trained_weights.pickle')


if __name__ == '__main__':
    main()
