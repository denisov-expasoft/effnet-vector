import numpy as np
import tensorflow as tf

from emulator.common.registry import Registry


def _np_half_up(data: np.ndarray) -> np.ndarray:
    return np.floor(data + 0.5)


def _np_half_up_symmetrical(data: np.ndarray) -> np.ndarray:
    data_sign = np.sign(data)  # pylint: disable=assignment-from-no-return
    data = np.abs(data) + 0.5

    return data_sign * np.floor(data)


def _tf_half_up(data: tf.Tensor, name: str = 'half_up_rounding') -> tf.Tensor:
    with tf.name_scope(name):
        return tf.floor(data + tf.constant(0.5, dtype=data.dtype))


def _tf_half_away(data: tf.Tensor, name: str = 'half_away_rounding') -> tf.Tensor:
    with tf.name_scope(name):
        data_sign = tf.sign(data)
        data = tf.abs(data) + 0.5

        return data_sign * tf.floor(data)


NP_ROUNDING_REGISTRY = Registry()
NP_ROUNDING_REGISTRY['floor'] = np.floor
NP_ROUNDING_REGISTRY['ceil'] = np.ceil
NP_ROUNDING_REGISTRY['nearest'] = np.round
NP_ROUNDING_REGISTRY['half-up'] = _np_half_up
NP_ROUNDING_REGISTRY['half-away'] = _np_half_up_symmetrical

TF_ROUNDING_REGISTRY = Registry()
TF_ROUNDING_REGISTRY['floor'] = tf.floor
TF_ROUNDING_REGISTRY['ceil'] = tf.ceil
TF_ROUNDING_REGISTRY['nearest'] = tf.round
TF_ROUNDING_REGISTRY['half-up'] = _tf_half_up
TF_ROUNDING_REGISTRY['half-away'] = _tf_half_away
