__all__ = [
    'central_crop',
    'aspect_preserving_resize',
]

import tensorflow as tf


def _get_central_crop_offset(size, target_size):
    return (size - target_size) // 2


def central_crop(image, target_height, target_width):
    shape = tf.shape(image)
    height, width = shape[0], shape[1]

    return tf.image.crop_to_bounding_box(
        image,
        _get_central_crop_offset(height, target_height),
        _get_central_crop_offset(width, target_width),
        target_height,
        target_width,
    )


def aspect_preserving_resize(image, target_height, target_width, image_fraction=None):
    shape = tf.shape(image)

    height, width = shape[0], shape[1]
    height_as_float, width_as_float = tf.cast(height, tf.float32), tf.cast(width, tf.float32)

    if image_fraction is not None:
        fraction_as_float = tf.cast(image_fraction, tf.float32)

        height_as_float = height_as_float * fraction_as_float
        width_as_float = width_as_float * fraction_as_float

        frac_height = tf.cast(height_as_float, tf.int32)
        frac_width = tf.cast(width_as_float, tf.int32)

        image = central_crop(image, frac_height, frac_width)

    target_height_as_float = tf.cast(target_height, tf.float32)
    target_width_as_float = tf.cast(target_width, tf.float32)

    scale_h = target_height_as_float / height_as_float
    scale_w = target_width_as_float / width_as_float
    scale = tf.maximum(scale_h, scale_w)

    new_height, new_width = tf.round(scale * height_as_float), tf.round(scale * width_as_float)
    new_height, new_width = tf.cast(new_height, tf.int32), tf.cast(new_width, tf.int32)

    resized_image = tf.image.resize_images(
        image,
        (new_height, new_width),
        method=tf.image.ResizeMethod.BILINEAR,
        align_corners=False,
    )
    cropped_image = central_crop(resized_image, target_height, target_width)

    return cropped_image
