from typing import Any
from typing import Dict
from typing import List

from emulator.ops_counter.layer_configuration_containers import *
import numpy as np

from emulator.ops_counter.count_tflite_like_ops import count_tflite_like_ops


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
    elif isinstance(op, Swish):
        in_channels = op.input_shape[-1]
        out_channels = op.input_shape[-1]
        input_size = op.input_shape[0]
    elif isinstance(op, Sigmoid):
        in_channels = op.input_shape[-1]
        out_channels = op.input_shape[-1]
        input_size = op.input_shape[0]
    elif isinstance(op, Mul):
        in_channels = op.input_shape[-1]
        out_channels = op.input_shape[-1]
        input_size = op.input_shape[0]
    else:
        raise ValueError('Encountered unknown operation %s.' % str(op))
    return input_size, kernel_size, in_channels, out_channels


class MicroNetCounter(object):
    """Counts operations using given information.

    """
    _header_str = '{:25} {:>10} {:>13} {:>13} {:>13} {:>15} {:>10} {:>10} {:>10}'
    _line_str = ('{:25s} {:10d} {:13d} {:13d} {:13d} {:15.3f} {:10.3f}'
                 ' {:10.3f} {:10.3f}')

    def __init__(self, all_ops, add_bits_base=32, mul_bits_base=32):
        self.all_ops = all_ops
        # Full precision add is counted one.
        self.add_bits_base = add_bits_base
        # Full precision multiply is counted one.
        self.mul_bits_base = mul_bits_base

    def _aggregate_list(self, counts):
        return np.array(counts).sum(axis=0)

    def process_counts(self, total_params, total_mults, total_adds):
        # converting to Mbytes.
        total_params = int(total_params) / 1e6
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
            param_count, flop_mults, flop_adds = count_tflite_like_ops(op_template)

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
            param_count, flop_mults, flop_adds = count_tflite_like_ops(op_template)

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

    def get_statistics(self) -> List[Dict[str, Any]]:
        """Collect info for all operations with given options."""

        ops_statistics = []

        # Let's count starting from zero.
        total_params, total_mults, total_adds = [0] * 3
        for op_name, op_template in self.all_ops:
            param_count, flop_mults, flop_adds = count_tflite_like_ops(op_template)

            temp_res = get_info(op_template)
            input_size, kernel_size, in_channels, out_channels = temp_res
            # At this point param_count, flop_mults, flop_adds should be read.
            total_params += param_count
            total_mults += flop_mults
            total_adds += flop_adds

            ops_statistics.append({
                'op_name': op_name,
                'inp_size': input_size,
                'kernel_size': kernel_size,
                'in_chs': in_channels,
                'out_chs': out_channels,
                'params': param_count,
                'muls': flop_mults,
                'adds': flop_adds,
                'flops': flop_mults + flop_adds,
            })

        return ops_statistics
