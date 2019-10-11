import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from emulator.common import GraphWalker
from emulator.common import LayerConfigurationParameters as Lcp
from emulator.common import SupportedLayerTypes as Slt
from emulator.ops_counter import layer_configuration_containers as cfg_container
from emulator.ops_counter.counter import MicroNetCounter
from emulator.ops_counter.ops_from_config import get_countable_ops
from emulator.ops_counter.ops_from_config import layer_to_op_info

_INPUT_IMG_SIZE = 224


def _add_input_shape(config, input_shape):
    config['input_node_1']['shape'] = input_shape

    output_shapes = {}

    graph_walker = GraphWalker(config)

    for layer_name, _, _ in graph_walker.walk_forward_iter():
        layer_cfg = config[layer_name]
        layer_type = Slt(layer_cfg[Lcp.ATTR_COMMON_TYPE.value])

        if layer_type is Slt.LAYER_INPUT:
            output_shapes[layer_name] = deepcopy(layer_cfg[Lcp.ATTR_IO_SHAPE.value])

        elif layer_type is Slt.LAYER_OUTPUT:
            continue

        elif layer_type in [Slt.LAYER_CONV2D, Slt.LAYER_CONV2D_DEPTHWISE, Slt.LAYER_FC]:
            input_layers = layer_cfg[Lcp.ATTR_COMMON_BOTTOM.value]
            layer_cfg[Lcp.ATTR_INPUT_SHAPE.value] = deepcopy(output_shapes[input_layers[0]])
            op_info: cfg_container.MatrixOps = layer_to_op_info(layer_cfg, config)
            output_shapes[layer_name] = op_info.output_shape

        elif layer_type in [Slt.LAYER_ADD, Slt.LAYER_SIGMOID, Slt.LAYER_SWISH]:
            input_layers = layer_cfg[Lcp.ATTR_COMMON_BOTTOM.value]
            layer_cfg[Lcp.ATTR_INPUT_SHAPE.value] = deepcopy(output_shapes[input_layers[0]])
            output_shapes[layer_name] = deepcopy(layer_cfg[Lcp.ATTR_INPUT_SHAPE.value])

        elif layer_type is Slt.LAYER_MUL:
            input_layers = layer_cfg[Lcp.ATTR_COMMON_BOTTOM.value]
            # Broadcasting shapes (using numpy broadcasting to calculate the shape)
            inp_shape_1, inp_shape_2 = output_shapes[input_layers[0]], output_shapes[input_layers[1]]
            inp_shape = inp_shape_1[:1] + list(np.shape(
                np.ones(inp_shape_1[1:]) * np.ones(inp_shape_2[1:])
            ))
            layer_cfg[Lcp.ATTR_INPUT_SHAPE.value] = deepcopy(inp_shape)
            output_shapes[layer_name] = deepcopy(layer_cfg[Lcp.ATTR_INPUT_SHAPE.value])

        elif layer_type is Slt.LAYER_REDUCE_MEAN:
            input_layers = layer_cfg[Lcp.ATTR_COMMON_BOTTOM.value]
            layer_cfg[Lcp.ATTR_INPUT_SHAPE.value] = deepcopy(output_shapes[input_layers[0]])

            keep_dims = layer_cfg[Lcp.ATTR_REDUCTION_KEEPDIMS.value]
            reduction_axis = layer_cfg[Lcp.ATTR_AXIS.value]
            if not isinstance(reduction_axis, list):
                reduction_axis = [reduction_axis]
            reduction_axis = [
                dim if dim > 0 else dim + len(input_shape)
                for dim in reduction_axis
            ]

            output_shape = [
                1 if i in reduction_axis else dim
                for i, dim in enumerate(input_shape)
                if keep_dims or i not in reduction_axis
            ]

            output_shapes[layer_name] = output_shape

        else:
            raise NotImplementedError(
                f'Shape prediction for layer type "{layer_cfg[Lcp.ATTR_COMMON_TYPE.value]}" is not implemented'
            )


def main():

    # with Path('model-data/regular.json').open('r') as file:
    with Path('model-data/fakequant.json').open('r') as file:
        config = json.load(file)

    _add_input_shape(config, input_shape=[None, _INPUT_IMG_SIZE, _INPUT_IMG_SIZE, 3])

    ops_ = get_countable_ops(config)
    counter_ = MicroNetCounter(all_ops=ops_)
    counter_.print_summary()
    total_params, total_muls, total_adds = counter_.process_counts(*counter_.total())
    final_score = total_params / 6.9 + (total_adds + total_muls) / 1170
    print(f'Score is {total_params:.2f} / 6.9 + {(total_adds + total_muls):.1f} / 1170 = {final_score:.4f}')

    # import pickle
    #
    # with Path('./model-data/ops-statistics.pickle').open('wb') as file:
    #     pickle.dump(counter_.get_statistics(), file)


if __name__ == '__main__':
    main()
