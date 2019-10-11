__all__ = [
    'ImagenetPreprocessing',
]

import tensorflow as tf

from emulator.common.data_types import TPyTransform
from emulator.dataset.base.dataset_transformation import BaseDatasetTransformation
from emulator.dataset.imagenet.preprocess_image import aspect_preserving_resize

_DEFAULT_IMAGE_FRACTION = 1.0


class ImagenetPreprocessing(BaseDatasetTransformation):

    def __init__(
            self,
            output_image_size: int,
            *,
            image_fraction: float = _DEFAULT_IMAGE_FRACTION,
            preprocess_fun: TPyTransform = None,
            num_parallel_calls: int = None,
    ):
        self._output_image_size = output_image_size
        if image_fraction is None:
            image_fraction = _DEFAULT_IMAGE_FRACTION
        self._image_fraction = image_fraction
        super().__init__(
            preprocess_fun=preprocess_fun,
            num_parallel_calls=num_parallel_calls,
        )

    def _prepare_data(self, data: tf.Tensor) -> tf.Tensor:
        image = tf.image.decode_image(data, channels=3)
        image.set_shape((None, None, 3))
        image = aspect_preserving_resize(
            image,
            self._output_image_size,
            self._output_image_size,
            self._image_fraction,
        )

        return super()._prepare_data(image)
