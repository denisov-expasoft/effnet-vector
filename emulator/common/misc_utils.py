__all__ = [
    'get_quant_limits',
    'normalize_to_list',
    'drop_nones',
    'any_tensor',
]

from typing import Any
from typing import List
from typing import Tuple

import tensorflow as tf


def get_quant_limits(bits_count: int, is_signed: bool) -> Tuple[int, int]:
    if is_signed:
        bound = 2 ** (bits_count - 1)
        return -bound, bound - 1

    return 0, 2 ** bits_count - 1


def normalize_to_list(data: Any) -> List[Any]:
    if data is None:
        return []

    if isinstance(data, list):
        return data

    return [data]


def drop_nones(data: dict) -> dict:
    return {
        key: value
        for key, value in data.items()
        if value is not None
    }


def get_dict_keys_by_value(data: dict, ref_value: Any) -> list:
    return [key for key, value in data.items() if value == ref_value]


def any_tensor(*parameters) -> bool:
    return any(isinstance(param, tf.Tensor) for param in parameters)
