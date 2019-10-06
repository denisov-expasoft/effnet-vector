__all__ = [
    'DEFAULT_OUTPUT_IMAGE_SIZE',
    'batch_stream_from_records',
]

from pathlib import Path
from typing import Callable
from typing import List
from typing import Union

import numpy as np

from emulator.dataset.base.batch_stream import BaseBatchStream
from emulator.dataset.base.batch_stream import RecordsWithLabelsBatchStream
from emulator.dataset.imagenet.dataset_transformation import ImagenetPreprocessing

DEFAULT_OUTPUT_IMAGE_SIZE = 224


def _default_preprocess_function(image: np.ndarray) -> np.ndarray:
    return image / 127.5 - 1.0


def batch_stream_from_records(
        records_paths: Union[Path, List[Path]],
        batch_size: int, *,
        output_image_size: int = DEFAULT_OUTPUT_IMAGE_SIZE,
        preprocess_fun: Callable[[np.ndarray], np.ndarray] = _default_preprocess_function,
        crop_fraction: float = None,
        num_parallel_calls: int = None,
) -> BaseBatchStream:
    imagenet_preprocessing = ImagenetPreprocessing(
        output_image_size,
        preprocess_fun=preprocess_fun,
        image_fraction=crop_fraction,
        num_parallel_calls=num_parallel_calls,
    )

    return RecordsWithLabelsBatchStream(
        records_paths,
        'image_encoded',
        'label',
        batch_size,
        transform_dataset_fun=imagenet_preprocessing,
    )
