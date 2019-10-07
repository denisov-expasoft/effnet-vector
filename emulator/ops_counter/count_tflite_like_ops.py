from typing import Tuple
from typing import Union

import numpy as np

from emulator.ops_counter.layer_configuration_containers import Add
from emulator.ops_counter.layer_configuration_containers import BaseOperation
from emulator.ops_counter.layer_configuration_containers import Conv2D
from emulator.ops_counter.layer_configuration_containers import DepthWiseConv2D
from emulator.ops_counter.layer_configuration_containers import FullyConnected
from emulator.ops_counter.layer_configuration_containers import GlobalAvg
from emulator.ops_counter.layer_configuration_containers import MatrixOps
from emulator.ops_counter.layer_configuration_containers import Mul
from emulator.ops_counter.layer_configuration_containers import Sigmoid
from emulator.ops_counter.layer_configuration_containers import Swish

# If we use strict accumulator we have to fix number of bits for every accumulator
_ACCUMULATOR_BITS = 32
_FULL_OPERATION_BITS = 32


def _count_matrix_ops_quantized(op: MatrixOps) -> Tuple[float, float, float]:
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
    number_of_adds = 0
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
    number_of_adds += out_vals_num * (params_per_single_output - 1) * mat_add_ops_cost

    # Count parameters
    # (zero point for quantized inputs is taken into account by the previous layer)

    # kernel size + `c_out` parameters for storing the weights zero point (vector)
    number_of_parameters = (np.prod(op.kernel_shape) + op.output_chs_num) * op.weight_bits / _FULL_OPERATION_BITS
    # zero point for quantized outputs
    number_of_parameters += op.output_bits / _FULL_OPERATION_BITS
    # Bias (assuming there are always a 32-bit bias)
    number_of_parameters += op.kernel_shape[-1]
    # 32-bit requantization fixed-point scale (unique for each output channel)
    number_of_parameters += 1 * op.output_chs_num

    # Count requantization costs

    # sum(q_a_i)
    number_of_adds += (
            (params_per_single_output - 1) *  # number of additions
            op.output_size ** 2  # `sum(q_a_i)` is unique for each spatial coordinate of the calculated output tensor
    ) * op.input_bits[0] / _FULL_OPERATION_BITS

    number_of_adds += (
            3 +  # (32-bit) [ N * z_a * z_w + q_b - z_a * sum(q_w_i) ] + [ z_w * sum(q_a_i) ] + [ sum(q_a_i * q_w_i) ]
            op.output_bits / _FULL_OPERATION_BITS  # [[ z_out + ... ]]
    ) * out_vals_num

    # Estimate number of bits necessary for the container for `sum(q_a_i)` operation
    block_sum_acc_bits = op.input_bits[0] + int(np.ceil(np.log2(params_per_single_output)))

    number_of_muls += 1 * out_vals_num  # 32-bit requantization[[ M * ... ]]

    # z_w * sum(q_a_i) is also unique for spatial coordinates only
    number_of_muls += op.output_size ** 2 * max(op.weight_bits, block_sum_acc_bits) / _FULL_OPERATION_BITS

    # Take the activation function into account
    if op.activation is not None:
        if op.activation == 'relu':
            number_of_muls += out_vals_num
        elif op.activation == 'sigmoid':
            number_of_muls += 2 * out_vals_num
            number_of_adds += 1 * out_vals_num
        elif op.activation == 'swish':
            number_of_muls += 3 * out_vals_num
            number_of_adds += 1 * out_vals_num
        else:
            raise ValueError(f'Unsupported activation function "{op.activation}"')

    return number_of_parameters, number_of_muls, number_of_adds


def _count_matrix_ops_float(op: MatrixOps) -> Tuple[float, float, float]:

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

    number_of_muls = out_vals_num * params_per_single_output
    number_of_adds = out_vals_num * (params_per_single_output - 1)
    number_of_parameters = int(np.prod(op.kernel_shape))

    if op.use_bias:
        number_of_adds += out_vals_num
        number_of_parameters += op.output_chs_num

    # Take the activation function into account
    if op.activation is not None:
        if op.activation == 'relu':
            number_of_muls += out_vals_num
        elif op.activation == 'sigmoid':
            number_of_muls += 2 * out_vals_num
            number_of_adds += 1 * out_vals_num
        elif op.activation == 'swish':
            number_of_muls += 3 * out_vals_num
            number_of_adds += 1 * out_vals_num
        else:
            raise ValueError(f'Unsupported activation function "{op.activation}"')


    return number_of_parameters, number_of_muls, number_of_adds


def _count_add_ops(op: Add) -> Tuple[float, float, float]:
    out_val_num = op.input_size * op.input_size * op.n_channels

    if op.quantized:
        number_of_parameters = 0
        number_of_parameters += op.output_bits / _FULL_OPERATION_BITS  # Output's zero point
        number_of_parameters += 3  # 32-bi requantization factors for two inputs and the output
        # Zero points for inputs are taken into account as the output zero points of the previous layers.

        # requantization of two input tensors, and requantization of the output
        number_of_muls = out_val_num * 3

        # Addition is performed using 32-bit values
        number_of_adds = out_val_num

    else:
        number_of_parameters = 0
        number_of_muls = 0
        number_of_adds = out_val_num

    return number_of_parameters, number_of_muls, number_of_adds


def _count_avg_ops(op: GlobalAvg) -> Tuple[float, float, float]:
    out_val_num = op.input_size * op.input_size * op.n_channels

    if op.quantized:
        # Global Average pooling is implemented as summation over the spatial
        # Zero point is the same for the input and output
        # (this parameter is taken into account as the output zero point of the previous layer)
        number_of_parameters = 1  # 32-bit requantization factor for the output

        # Number of additions is one less than the number of arguments
        number_of_adds = (op.input_size ** 2 - 1) * op.n_channels * op.input_bits[0] / _FULL_OPERATION_BITS
        number_of_muls = op.n_channels  # requantization with a 32-bit fixed-point scale

    else:
        number_of_parameters = 0
        number_of_muls = op.n_channels
        number_of_adds = (op.input_size ** 2 - 1) * op.n_channels

    return number_of_parameters, number_of_muls, number_of_adds


def _count_sigmoid_based_ops(op: Union[Sigmoid, Swish]) -> Tuple[float, float, float]:
    out_val_num = int(np.prod(op.input_shape))

    if isinstance(op, Swish):
        number_of_muls = 3 * out_val_num
    else:
        number_of_muls = 2 * out_val_num

    number_of_adds = 1 * out_val_num

    if op.quantized:
        number_of_parameters = 1  # 32-bit requantization factor for the output
        number_of_muls += 1 * out_val_num  # output requantization using 32-bit fixed-point scale
    else:
        number_of_parameters = 0

    return number_of_parameters, number_of_muls, number_of_adds


def _count_mul_layer_ops(op: Mul) -> Tuple[float, float, float]:
    out_val_num = int(np.prod(op.input_shape))

    # Possible variants:
    # 1) out_32 = M * (inp1_q - inp1_zero_q) * (inp2_q - inp2_zero_q)
    # 2) out_q = out_zero_q + M * (inp1_q - inp1_zero_q) * (inp2_q - inp2_zero_q)
    # 3) out_32 = M * (inp1_q - inp1_zero_q) * inp2_32
    # 4) out_q = out_zero_q + M * (inp1_q - inp1_zero_q) * inp2_32
    # 5) out_32 = inp1_q * inp2_32

    if not op.quantized and not op.output_quantized:
        rescale_cost = 0
        number_of_parameters = 0
    else:
        number_of_parameters = 1
        rescale_cost = 1

    if op.input_bits[0] < 16:
        add_1_cost = op.input_bits[0] / _FULL_OPERATION_BITS
        add_1_container_bits = op.input_bits[0] + 1
    else:
        add_1_cost = 0  # input_1 is already in float32
        add_1_container_bits = 32

    if op.input_bits[1] < 16:
        add_2_cost = op.input_bits[0] / _FULL_OPERATION_BITS
        add_2_container_bits = op.input_bits[0] + 1
    else:
        add_2_cost = 0 # input_2 is already in float32
        add_2_container_bits = 32

    if op.output_quantized:
        add_out_cost = op.output_bits / _FULL_OPERATION_BITS
        number_of_parameters += op.output_bits / _FULL_OPERATION_BITS  # output zero point
    else:
        add_out_cost = 0

    mul_cost = max(add_1_container_bits, add_2_container_bits) / _FULL_OPERATION_BITS

    number_of_adds = (add_1_cost + add_2_cost + add_out_cost) * out_val_num
    number_of_muls = (
            mul_cost * out_val_num +  # multiplication of inputs
            rescale_cost * out_val_num  # optional rescaling
    )

    return number_of_parameters, number_of_muls, number_of_adds


def count_tflite_like_ops(op: BaseOperation) -> Tuple[float, float, float]:

    if isinstance(op, MatrixOps):
        if op.quantized:
            return _count_matrix_ops_quantized(op)
        return  _count_matrix_ops_float(op)

    if isinstance(op, Add):
        return _count_add_ops(op)

    if isinstance(op, GlobalAvg):
        return _count_avg_ops(op)

    if isinstance(op, (Sigmoid, Swish)):
        return _count_sigmoid_based_ops(op)

    if isinstance(op, Mul):
        return _count_mul_layer_ops(op)

    raise NotImplementedError(
        f'Missing counter for the operation type "{op.__class__.__name__}"'
    )
