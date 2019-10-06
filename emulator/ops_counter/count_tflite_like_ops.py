from typing import Tuple

import numpy as np

from emulator.ops_counter.layer_configuration_containers import Add
from emulator.ops_counter.layer_configuration_containers import BaseOperation
from emulator.ops_counter.layer_configuration_containers import Conv2D
from emulator.ops_counter.layer_configuration_containers import DepthWiseConv2D
from emulator.ops_counter.layer_configuration_containers import FullyConnected
from emulator.ops_counter.layer_configuration_containers import GlobalAvg
from emulator.ops_counter.layer_configuration_containers import MatrixOps

# If we use strict accumulator we have to fix number of bits for every accumulator
_ACCUMULATOR_BITS = 32
_FULL_OPERATION_BITS = 32


def _count_matrix_ops(op: MatrixOps) -> Tuple[float, float, float]:
    # We count the number of operations based on the paper
    # http://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf
    # with some slight changes:
    #
    # Calculation of a single output value:
    # q_out = (
    #         z_out +                           -> One quantized parameter + one quantized op per each output value
    #         M * (                             -> Requantize using a 32-bit fixed-point scale (whole op + one param)
    #                 N * z_a * z_w + q_b -     -> Precalculated new 32-bit bias: whole op + 32-bit param
    #                 z_a * sum(q_w_i) -           (weights are also fixed and summation can be performed in advance).
    #                 z_w * sum(q_a_i) -        -> One reduced precision multiplication + N addtions of quantized values
    #                 sum(q_a_i * q_w_i)        -> Matrix multiplication (depends on the type of an operation)
    #         )
    # )

    mul_ops_cost = max(op.input_bits[0], op.weight_bits) / _FULL_OPERATION_BITS
    mat_add_ops_cost = (op.input_bits[0] + op.weight_bits) / _FULL_OPERATION_BITS

    # Counting operations
    number_of_ads = 0
    number_of_muls = 0

    if isinstance(op, Conv2D):
        out_vals_num = op.output_size ** 2 * op.output_chs_num  # number of output values (batch size is 1)
        k_size = op.kernel_shape[0]
        params_per_single_output = k_size ** 2 * op.input_chs_num

    elif isinstance(op, DepthWiseConv2D):
        out_vals_num = op.output_size ** 2 * op.output_chs_num  # number of output values (batch size is 1)
        k_size = op.kernel_shape[0]
        params_per_single_output = k_size ** 2

    elif isinstance(op, FullyConnected):
        out_vals_num = op.output_chs_num  # number of output values (batch size is 1)
        params_per_single_output = op.input_chs_num

    else:
        raise TypeError(f'Counting for subclass {op.__class__.__name__} of MatrixOps is not implemented')

    number_of_muls += out_vals_num * params_per_single_output * mul_ops_cost
    number_of_ads += out_vals_num * (params_per_single_output - 1) * mat_add_ops_cost

    # Count parameters
    # (zero point for quantized inputs is taken into account by the previous layer)

    # kernel size + one parameter for storing the weights zero point
    number_of_parameters = (np.prod(op.kernel_shape) + 1) * op.weight_bits / _FULL_OPERATION_BITS
    # zero point for quantized outputs
    number_of_parameters += op.output_bits / _FULL_OPERATION_BITS
    # Bias (assuming there are always a 32-bit bias)
    number_of_parameters += op.kernel_shape[-1]
    # 32-bit requantization fixed-poin scale
    number_of_parameters += 1

    # Count requantization costs

    # sum(q_a_i)
    number_of_ads += (
            (params_per_single_output - 1) *  # number of additions
            op.output_size ** 2  # `sum(q_a_i)` is unique for each spatial coordinate of the calculated output tensor
    ) * op.input_bits[0] / _FULL_OPERATION_BITS

    number_of_ads += (
            3 +  # (32-bit) [ N * z_a * z_w + q_b - z_a * sum(q_w_i) ] + [ z_w * sum(q_a_i) ] + [ sum(q_a_i * q_w_i) ]
            op.output_bits / _FULL_OPERATION_BITS  # [[ z_out + ... ]]
    ) * out_vals_num

    # Estimate number of bits necessary for the container for `sum(q_a_i)` operation
    block_sum_acc_bits = op.input_bits[0] + int(np.ceil(np.log2(params_per_single_output)))

    number_of_muls += 1 * out_vals_num  # 32-bit requantization[[ M * ... ]]

    # z_w * sum(q_a_i) is also unique for spatial coordinates only
    number_of_muls += op.output_size ** 2 * max(op.weight_bits, block_sum_acc_bits) / _FULL_OPERATION_BITS

    # Take into accoun the activation function
    if op.activation is not None:
        if op.activation != 'relu':
            raise ValueError(f'Unsupported activation function "{op.activation}"')
        number_of_muls += out_vals_num

    return number_of_parameters, number_of_muls, number_of_ads


def count_tflite_like_ops(op: BaseOperation) -> Tuple[float, float, float]:

    if isinstance(op, MatrixOps):
        return _count_matrix_ops(op)

    if isinstance(op, Add):
        number_of_parameters = 0
        number_of_parameters += op.output_bits / _FULL_OPERATION_BITS  # Output's zero point
        number_of_parameters += 3  # 32-bi requantization factors for two inputs and the output
        # Zero points for inputs are taken into account as the output zero points of the previous layers.

        # requantization of two input tensors, and requantization of the output
        number_of_muls = op.input_size * op.input_size * op.n_channels * 3

        # Addition is performed using 32-bit values
        number_of_ads = op.input_size * op.input_size * op.n_channels

        return number_of_parameters, number_of_muls, number_of_ads

    if isinstance(op, GlobalAvg):
        # Global Average pooling is implemented as summation over the spatial
        # Zero point is the same for the input and output
        # (this parameter is taken into account as the output zero point of the previous layer)
        number_of_parameters = 1  # 32-bi requantization factor for the output

        # Number of additions is one less than the number of arguments
        number_of_ads = (op.input_size ** 2 - 1) * op.n_channels * op.input_bits[0] / _FULL_OPERATION_BITS
        number_of_muls = op.n_channels  # requantization with a 32-bit fixed-point scale

        return number_of_parameters, number_of_muls, number_of_ads

    raise NotImplementedError(
        f'Missing counter for the operation type "{op.__class__.__name__}"'
    )
