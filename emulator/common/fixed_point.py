__all__ = [
    'create_fixedpoint_scale',
    'find_optimal_fixed_point_position_for_array',
]

from typing import Tuple

import numpy as np


def _check_input_data(number: np.ndarray) -> None:
    any_nan = np.isnan(number).any()
    if any_nan:
        raise ValueError(
            'Input scale contains NaN'
        )

    any_inf = np.isinf(number).any()
    if any_inf:
        raise ValueError(
            'Scale must be finite'
        )

    any_neg = np.asarray(number <= 0).any()
    if any_neg:
        raise ValueError(
            'Scale must be greater than zero'
        )


def _find_integer_part_bits_number(number: np.ndarray) -> np.ndarray:
    # Calculate the number of bits, which can contain
    # the minimal integer number larger than the specified one
    number = np.asarray(number)
    number = number.astype(int)

    if np.all(number == 0):
        return np.array(0)

    n_bits = np.floor(np.log2(number)) + 1
    n_bits = np.asarray(n_bits, int)
    return n_bits


def create_fixedpoint_scale(
        number: np.ndarray,
        total_bit_width: int = 24,
) -> Tuple[np.ndarray, np.ndarray]:
    """"""
    _check_input_data(number)
    n_bits = _find_integer_part_bits_number(number)

    # We normalize the initial number to fit
    # the specified container for fixedpoint scale
    normalized_number = number * (2. ** (total_bit_width - n_bits))
    normalized_number = np.clip(normalized_number, 0, 2 ** total_bit_width - 1)
    normalized_number = normalized_number.round().astype(int)

    max_2_pow = np.gcd(normalized_number, 2 ** total_bit_width)
    max_2_pow = np.log2(max_2_pow).astype(int)

    fixedpoint_scale = normalized_number // (2 ** max_2_pow)
    fixedpoint_scale = np.asarray(fixedpoint_scale, int)

    fixedpoint_shift = max_2_pow - (total_bit_width - n_bits)
    fixedpoint_shift = np.asarray(fixedpoint_shift, int)

    return fixedpoint_scale, fixedpoint_shift


def find_optimal_fixed_point_position_for_array(
        number: np.ndarray,
        total_bit_width: int = 24,
) -> int:
    """"""
    _check_input_data(number)
    number = np.max(number)
    n_bits = _find_integer_part_bits_number(number)
    integer_max_bits = int(np.clip(n_bits, 0, total_bit_width))
    # the rest number of bits is for fractional part of number,
    # it defines position of the fixed point
    fixed_point_position = total_bit_width - integer_max_bits
    return fixed_point_position


if __name__ == '__main__':
    print('Shift, scale:', create_fixedpoint_scale(np.asarray(13 / 1780), 8))
    pass
