from emulator.ops_counter.layer_configuration_containers import *
import numpy as np
from typing import List
from typing import Tuple

_ACCUMULATOR_BITS = 24
_STRICT_ACCUMULATOR = False


def get_conv_output_size(
        image_size: int,
        filter_size: int,
        padding: str,
        stride: int,
) -> int:
    """Calculates the output size of convolution.

    The input, filter and the strides are assumed to be square.
    Arguments:
      image_size: int, Dimensions of the input image (square assumed).
      filter_size: int, Dimensions of the kernel (square assumed).
      padding: str, padding added to the input image. 'same' or 'valid'
      stride: int, stride with which the kernel is applied (square assumed).
    Returns:
      int, output size.
    """
    if padding == 'same':
        pad = filter_size // 2
    elif padding == 'valid':
        pad = 0
    else:
        raise NotImplementedError(
            f'Padding: {padding} should be `same` or `valid`.'
        )
    out_size = np.ceil((image_size - filter_size + 1. + 2 * pad) / stride)
    return int(out_size)


def get_sparse_size(
        tensor_shape: List[int],
        param_bits: int,
        sparsity: float,
) -> int:
    """Given a tensor shape returns #bits required to store the tensor sparse.

    If sparsity is greater than 0, we do have to store a bit mask to represent
    sparsity.
    Args:
      tensor_shape: list<int>, shape of the tensor
      param_bits: int, number of bits the elements of the tensor represented in.
      sparsity: float, sparsity level. 0 means dense.
    Returns:
      int, number of bits required to represented the tensor in sparse format.
    """
    n_elements = np.prod(tensor_shape)
    c_size = n_elements * param_bits * (1 - sparsity)
    if sparsity > 0:
        c_size += n_elements  # 1 bit binary mask.
    return c_size


def get_flops_per_activation(activation: str) -> Tuple[int, int]:
    """Returns the number of multiplication ands additions of an activation.

    Args:
      activation: str, type of activation applied to the output.
    Returns:
      n_muls, n_adds
    """
    activation = activation.lower()
    if activation == 'relu':
        # For the purposes of the "freebie" quantization scoring, ReLUs can be
        # assumed to be performed on 16-bit inputs. Thus, we track them as
        # multiplications in our accounting, which can also be assumed to be
        # performed on reduced precision inputs.
        return 1, 0
    elif activation == 'swish':  # Swish: x / (1 + exp(-bx))
        return 3, 1
    elif activation == 'sigmoid':  # Sigmoid: exp(x) / (1 + exp(x))
        return 2, 1
    elif activation == 'relu6':
        return 2, 0
    else:
        raise ValueError(f'activation: {activation} is not valid')


def count_tflite_like_ops(op: BaseOperation, param_bits: int = 32):
    # consider that sparsity is always 0
    param_count = 0
    if isinstance(op, Conv2D):
        k_size, _, c_in, c_out = op.kernel_shape
        filter_size = k_size*k_size*c_in
        output_shape = get_conv_output_size(op.input_size, k_size, op.padding, op.strides[0]) ** 2 * c_out
        param_count += get_sparse_size([k_size, k_size, c_in, c_out], op.weight_bits, 0)
        op_bits = max(max(op.input_bits), op.weight_bits)
        op_cost = op_bits / param_bits
        param_count += c_out*param_bits + c_out * op.weight_bits  # scale and shift

    elif isinstance(op, DepthWiseConv2D):
        k_size, _, c_in, c_out = op.kernel_shape
        filter_size = k_size*k_size
        output_shape = get_conv_output_size(op.input_size, k_size, op.padding, op.strides[0]) ** 2 * c_out
        param_count += get_sparse_size([k_size, k_size, c_in, c_out], op.weight_bits, 0)
        op_bits = max(max(op.input_bits), op.weight_bits)
        op_cost = op_bits / param_bits
        param_count += c_out*param_bits + c_out*op.weight_bits  # scale and shift

    elif isinstance(op, FullyConnected):
        num_in, num_out = op.kernel_shape
        filter_size = num_in
        output_shape = num_out
        param_count += get_sparse_size([num_in, num_out], op.weight_bits, 0)
        op_bits = max(max(op.input_bits), op.weight_bits)
        op_cost = op_bits / param_bits
        param_count += num_out*param_bits + num_out*op.weight_bits  # scale and shift

    elif isinstance(op, Add):
        op_cost = max(op.input_bits)/param_bits
        num_of_muls = op.input_size * op.input_size * op.n_channels * 3  # scale and clip_by_value
        num_of_adds = (op.input_size * op.input_size * op.n_channels + 1)*op_cost  # add ops and shift
        param_count = op.n_channels + op.n_channels*op_cost  # Scale and shift
        return param_count*max(op.input_bits), num_of_muls, num_of_adds

    elif isinstance(op, GlobalAvg):
        op_cost = max(op.input_bits) / param_bits
        num_of_muls = op.n_channels  # we can precalculate scale/image_size*image_size
        # We have to add values over spatial dimensions.
        # For each value two adds. original add and shift
        num_of_adds = (op.input_size * op.input_size) * op.n_channels * 2 * op_cost
        param_count = op.n_channels + op.n_channels*op_cost  # scale and shift
        return param_count*max(op.input_bits), num_of_muls, num_of_adds

    else:
        raise NotImplementedError(
            f'Unsupported layer type {type(op)}'
        )

    # q_out = clip_by_value(M(N*z_a*z_w - z_w*sum(q_a_i) - z_a*sum(q_w_i) + sum(q_a_i*q_w_i) + q_b))
    #               2       1            1   1  N*op_cos 1                1 2*N*op_cost N*op_cost 1
    # we assume that after multiplication the container grows twice
    num_of_muls = (filter_size*op_cost + 4)*output_shape
    if _STRICT_ACCUMULATOR:
        scale = _ACCUMULATOR_BITS/16
    else:
        scale = 1
    if op.use_bias:
        num_of_adds = ((filter_size + filter_size*2*scale + 1)*op_cost + 3/scale)*output_shape + output_shape/scale
    else:
        num_of_adds = ((filter_size + filter_size*2*scale + 1)*op_cost + 3/scale)*output_shape

    if op.activation:
        n_muls, n_adds = get_flops_per_activation(op.activation)
        # all activations counts as full ops
        num_of_muls += n_muls * output_shape
        num_of_adds += n_adds * output_shape

    return param_count, num_of_muls, num_of_adds


def get_info(op: BaseOperation):
    """Given an op extracts some common information."""
    input_size, kernel_size, in_channels, out_channels = [-1] * 4
    if isinstance(op, (DepthWiseConv2D, Conv2D)):
        # square kernel assumed.
        kernel_size, _, in_channels, out_channels = op.kernel_shape
        input_size = op.input_size
    elif isinstance(op, GlobalAvg):
        in_channels = op.n_channels
        out_channels = 1
        input_size = op.input_size
    elif isinstance(op, Add):
        in_channels = op.n_channels
        out_channels = op.n_channels
        input_size = op.input_size
    elif isinstance(op, FullyConnected):
        in_channels, out_channels = op.kernel_shape
        input_size = 1
    elif isinstance(op, AvgPool):
        in_channels = op.input_channels
        out_channels = -1
        input_size = op.input_size
    elif isinstance(op, Swish):
        in_channels = -1
        out_channels = -1
        input_size = -1
    elif isinstance(op, Sigmoid):
        in_channels = -1
        out_channels = -1
        input_size = -1
    elif isinstance(op, Mul):
        in_channels = -1
        out_channels = -1
        input_size = -1
    else:
        raise ValueError('Encountered unknown operation %s.' % str(op))
    return input_size, kernel_size, in_channels, out_channels


def count_ops(op: BaseOperation, param_bits: int = 32):
    """Given a operation class returns the flop and parameter statistics.

    Args:
      op: BaseOperation, class container for operation
      param_bits: int, number of bits required to represent a parameter.
    Returns:
      param_count: number of bits required to store parameters
      n_mults: number of multiplications made per input sample.
      n_adds: number of multiplications made per input sample.
    """
    flop_mults = flop_adds = param_count = 0
    if isinstance(op, Conv2D):

        # Square kernel expected.
        assert op.kernel_shape[0] == op.kernel_shape[1]
        k_size, _, c_in, c_out = op.kernel_shape

        # Size of the possibly sparse convolutional tensor.
        param_count += get_sparse_size(
            [k_size, k_size, c_in, c_out], param_bits, 0)

        # Square stride expected.
        assert op.strides[0] == op.strides[1]
        stride = op.strides[0]

        # Each application of the kernel can be thought as a dot product between
        # the flattened kernel and patches of the image.
        vector_length = (k_size * k_size * c_in)
        # Number of elements in the output is OUT_SIZE * OUT_SIZE * OUT_CHANNEL
        n_output_elements = get_conv_output_size(op.input_size, k_size, op.padding,
                                                 stride) ** 2 * c_out
        # Each output is the product of a one dot product. Dot product of two
        # vectors of size n needs n multiplications and n - 1 additions.
        flop_mults += vector_length * n_output_elements
        flop_adds += vector_length * n_output_elements

        if op.use_bias:
            # For each output channel we need a bias term.
            param_count += c_out * param_bits
            # If we have bias we need one more addition per dot product.
            flop_adds += n_output_elements

        if op.activation:
            # We would apply the activaiton to every single output element.
            n_muls, n_adds = get_flops_per_activation(op.activation)
            flop_mults += n_muls * n_output_elements
            flop_adds += n_adds * n_output_elements

    elif isinstance(op, DepthWiseConv2D):
        # Square kernel expected.
        assert op.kernel_shape[0] == op.kernel_shape[1]
        # Last dimension of the kernel should be 1.
        assert op.kernel_shape[3] == 1
        k_size, _, channels, _ = op.kernel_shape

        # Size of the possibly sparse convolutional tensor.
        param_count += get_sparse_size(
            [k_size, k_size, channels], param_bits, 0)

        # Square stride expected.
        assert op.strides[0] == op.strides[1]
        stride = op.strides[0]

        # Each application of the kernel can be thought as a dot product between
        # the flattened kernel and patches of the image.
        vector_length = (k_size * k_size)
        # Number of elements in the output tensor.

        n_output_elements = get_conv_output_size(op.input_size, k_size, op.padding,
                                                 stride) ** 2 * channels

        # Each output is the product of a one dot product. Dot product of two
        # vectors of size n needs n multiplications and n - 1 additions.
        flop_mults += vector_length * n_output_elements
        flop_adds += vector_length * n_output_elements

        if op.use_bias:
            # For each output channel we need a bias term.
            param_count += channels * param_bits
            # If we have bias we need one more addition per dot product.
            flop_adds += n_output_elements

        if op.activation:
            # We would apply the activaiton to every single output element.
            n_muls, n_adds = get_flops_per_activation(op.activation)
            flop_mults += n_muls * n_output_elements
            flop_adds += n_adds * n_output_elements
    elif isinstance(op, GlobalAvg):
        # For each output channel we will make a division.
        flop_mults += op.n_channels
        # We have to add values over spatial dimensions.
        flop_adds += (op.input_size * op.input_size - 1) * op.n_channels
    elif isinstance(op, Add):
        # Number of elements many additions.
        flop_adds += op.input_size * op.input_size * op.n_channels
    elif isinstance(op, FullyConnected):
        c_in, c_out = op.kernel_shape
        # Size of the possibly sparse weight matrix.
        param_count += get_sparse_size(
            [c_in, c_out], param_bits, 0)

        # number of non-zero elements for the sparse dot product.
        n_elements = c_in
        flop_mults += n_elements * c_out
        # We have one less addition than the number of multiplications per output
        # channel.
        flop_adds += (n_elements - 1) * c_out

        if op.use_bias:
            param_count += c_out * param_bits
            flop_adds += c_out
        if op.activation:
            n_muls, n_adds = get_flops_per_activation(op.activation)
            flop_mults += n_muls * c_out
            flop_adds += n_adds * c_out
    elif isinstance(op, AvgPool):

        _, k_size, _, _ = op.kernel_shape
        vector_length = (k_size * k_size)
        # Number of elements in the output tensor.
        stride = op.strides[0]
        n_output_elements = get_conv_output_size(op.input_size, k_size, op.padding,
                                                 stride) ** 2 * op.input_channels

        # Each output is the product of a one dot product. Dot product of two
        # vectors of size n needs n multiplications and n - 1 additions.
        flop_mults += n_output_elements
        flop_adds += vector_length * n_output_elements
    elif isinstance(op, Swish):
        n_muls, n_adds = get_flops_per_activation(op.activation)
        n_output_elements = 1
        for dim in op.input_shape:
            n_output_elements *= dim
        flop_mults += n_muls * n_output_elements
        flop_adds += n_adds * n_output_elements
    elif isinstance(op, Sigmoid):
        n_muls, n_adds = get_flops_per_activation(op.activation)
        n_output_elements = 1
        for dim in op.input_shape:
            n_output_elements *= dim
        flop_mults += n_muls * n_output_elements
        flop_adds += n_adds * n_output_elements
    elif isinstance(op, Mul):
        n_output_elements = 1
        for dim in op.input_shape:
            n_output_elements *= dim
        flop_mults += n_output_elements
    else:
        raise ValueError('Encountered unknown operation %s.' % str(op))

    return param_count, flop_mults, flop_adds


class MicroNetCounter(object):
    """Counts operations using given information.

    """
    _header_str = '{:25} {:>10} {:>13} {:>13} {:>13} {:>15} {:>10} {:>10} {:>10}'
    _line_str = ('{:25s} {:10d} {:13d} {:13d} {:13d} {:15.3f} {:10.3f}'
                 ' {:10.3f} {:10.3f}')

    def __init__(self, all_ops, add_bits_base=32, mul_bits_base=32, is_quantized: bool = False):
        self.all_ops = all_ops
        # Full precision add is counted one.
        self.add_bits_base = add_bits_base
        # Full precision multiply is counted one.
        self.mul_bits_base = mul_bits_base

        self._is_quantized = is_quantized

    def _aggregate_list(self, counts):
        return np.array(counts).sum(axis=0)

    def process_counts(self, total_params, total_mults, total_adds):
        # converting to Mbytes.
        total_params = int(total_params) / 32. / 1e6
        total_mults = total_mults / 1e6
        total_adds = total_adds / 1e6
        return total_params, total_mults, total_adds

    def _print_header(self):
        output_string = self._header_str.format(
            'op_name', 'inp_size', 'kernel_size', 'in channels', 'out channels',
            'params(MBytes)', 'mults(M)', 'adds(M)', 'MFLOPS')
        print(output_string)
        print(''.join(['='] * 125))

    def _print_line(self, name, input_size, kernel_size, in_channels,
                    out_channels, param_count, flop_mults, flop_adds, base_str=None):
        """Prints a single line of operation counts."""
        op_pc, op_mu, op_ad = self.process_counts(param_count, flop_mults,
                                                  flop_adds)
        if base_str is None:
            base_str = self._line_str
        output_string = base_str.format(
            name, input_size, kernel_size, in_channels, out_channels, op_pc,
            op_mu, op_ad, op_mu + op_ad)
        print(output_string)

    def total(self):

        total_params, total_mults, total_adds = [0] * 3
        for op_name, op_template in self.all_ops:
            if self._is_quantized:
                param_count, flop_mults, flop_adds = count_tflite_like_ops(op_template)
            else:
                param_count, flop_mults, flop_adds = count_ops(op_template)

            total_params += param_count
            total_mults += flop_mults
            total_adds += flop_adds

        return total_params, total_mults, total_adds

    def print_summary(self):
        """Prints all operations with given options."""

        self._print_header()
        # Let's count starting from zero.
        total_params, total_mults, total_adds = [0] * 3
        for op_name, op_template in self.all_ops:
            if self._is_quantized:
                param_count, flop_mults, flop_adds = count_tflite_like_ops(op_template)
            else:
                param_count, flop_mults, flop_adds = count_ops(op_template)
            temp_res = get_info(op_template)
            input_size, kernel_size, in_channels, out_channels = temp_res
            # At this point param_count, flop_mults, flop_adds should be read.
            total_params += param_count
            total_mults += flop_mults
            total_adds += flop_adds
            # Print the operation.
            self._print_line(op_name, input_size, kernel_size, in_channels,
                             out_channels, param_count, flop_mults, flop_adds)

        # Print Total values.
        # New string since we are passing empty strings instead of integers.
        out_str = ('{:25s} {:10s} {:13s} {:13s} {:13s} {:15.3f} {:10.3f} {:10.3f} '
                   '{:10.3f}')
        self._print_line(
            'total', '', '', '', '', total_params, total_mults, total_adds, base_str=out_str)
