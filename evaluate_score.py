import json
import pickle
from pathlib import Path

import emulator
from emulator.ops_counter.counter import MicroNetCounter
from emulator.ops_counter.ops_from_config import get_countable_ops

_QUANT_CONFIG_PATH = Path('model-data/fakequant.json')
_REGULAR_CONFIG_PATH = Path('model-data/regular.json')
_WEIGHTS_PATH = Path('model-data/weights_rescaled.pickle')


def _add_input_shape(config, reg_config, weights):
    reg_net = emulator.RegularModel(reg_config, weights)
    layer_names = list(config.keys())
    for l_n in layer_names:
        if l_n == 'input_node_1':
            continue
        config[l_n]['input_shape'] = list(reg_net.get_layer_output_shape(config[l_n]['bottom'][0]))


def main():
    with _QUANT_CONFIG_PATH.open('r') as file:
        config = json.load(file)

    with _REGULAR_CONFIG_PATH.open('r') as file:
        reg_config = json.load(file)

    with _WEIGHTS_PATH.open('rb') as file:
        weights = pickle.load(file)
    reg_config['input_node_1']['shape'] = [None, 256, 256, 3]
    _add_input_shape(config, reg_config, weights)
    ops_ = get_countable_ops(config, True)
    counter_ = MicroNetCounter(all_ops=ops_, is_quantized=True)
    counter_.print_summary()
    total_params, total_muls, total_adds = counter_.process_counts(*counter_.total())
    final_score = total_params / 6.9 + (total_adds + total_muls) / 1170
    print(f'Score is {total_params:.2f} / 6.9 + {(total_adds + total_muls):.1f} / 1170 = {final_score:.4f}')


if __name__ == '__main__':
    main()
