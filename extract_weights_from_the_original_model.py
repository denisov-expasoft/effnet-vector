# Extracting weights from the original pb-file using a mapping dictionary
# Batch normalization is folding into weights of preceding layers

import json
import pickle
from collections import defaultdict
from pathlib import Path

import click
import numpy as np
import tensorflow as tf

_LAYER_TO_NODE_PATH = Path("./model-data/layer_map.json")


def strip_node_name(name: str) -> str:
    name = name.replace('^', '')
    name = name.split(':')[0]
    return name


class GraphMap:
    def __init__(self, graph_def: tf.GraphDef):
        self._name_to_node_def = dict()
        self._forward_map = defaultdict(list)
        self._graph_def = graph_def
        self._build()

    def _build(self):
        for node_def in self._graph_def.node:
            self._name_to_node_def[node_def.name] = node_def
            node_inputs = node_def.input
            for node in node_inputs:
                self._forward_map[strip_node_name(node)].append(node_def.name)

    def get_node_by_name(self, node_name):
        return self._name_to_node_def[node_name]

    def get_node_consumers_nodedef(self, node_name):
        return [self.get_node_by_name(out_node_name) for out_node_name in self._forward_map[node_name]]


def _raise_node_not_found():
    raise ValueError(f'node not found')


def get_layer_param(layer_name, graph_map):
    layer_param = dict()
    node_def = graph_map.get_node_by_name(layer_name)
    identity_node_def = graph_map.get_node_by_name(node_def.input[1])
    if identity_node_def.op != 'Identity':
        _raise_node_not_found()
    const_node_def = graph_map.get_node_by_name(identity_node_def.input[0])
    if const_node_def.op != 'Const':
        _raise_node_not_found()
    layer_param['weights'] = tf.make_ndarray(const_node_def.attr['value'].tensor)

    bias_add = graph_map.get_node_consumers_nodedef(layer_name)[0]
    if bias_add.op != 'BiasAdd':
        return layer_param

    identity_node_def = graph_map.get_node_by_name(bias_add.input[1])
    if identity_node_def.op != 'Identity':
        _raise_node_not_found()
    const_node_def = graph_map.get_node_by_name(identity_node_def.input[0])
    layer_param['bias'] = tf.make_ndarray(const_node_def.attr['value'].tensor)
    return layer_param


def get_batch_norm_param(layer_name, graph_map):
    # gamma
    bn_param = dict()
    node_def = graph_map.get_node_by_name(layer_name)
    identity_node_def = graph_map.get_node_by_name(node_def.input[1])
    if identity_node_def.op != 'Identity':
        _raise_node_not_found()

    const_node_def = graph_map.get_node_by_name(identity_node_def.input[0])
    if const_node_def.op != 'Const':
        _raise_node_not_found()

    bn_param['gamma'] = tf.make_ndarray(const_node_def.attr['value'].tensor)

    # beta
    node_def = graph_map.get_node_by_name(layer_name)
    identity_node_def = graph_map.get_node_by_name(node_def.input[2])
    if identity_node_def.op != 'Identity':
        _raise_node_not_found()

    const_node_def = graph_map.get_node_by_name(identity_node_def.input[0])
    if const_node_def.op != 'Const':
        _raise_node_not_found()

    bn_param['beta'] = tf.make_ndarray(const_node_def.attr['value'].tensor)

    # moving_mean
    node_def = graph_map.get_node_by_name(layer_name)
    identity_node_def = graph_map.get_node_by_name(node_def.input[3])
    if identity_node_def.op != 'Identity':
        _raise_node_not_found()

    const_node_def = graph_map.get_node_by_name(identity_node_def.input[0])
    if const_node_def.op != 'Const':
        _raise_node_not_found()

    bn_param['moving_mean'] = tf.make_ndarray(const_node_def.attr['value'].tensor)

    # moving_variance
    node_def = graph_map.get_node_by_name(layer_name)
    identity_node_def = graph_map.get_node_by_name(node_def.input[4])
    if identity_node_def.op != 'Identity':
        _raise_node_not_found()

    const_node_def = graph_map.get_node_by_name(identity_node_def.input[0])
    if const_node_def.op != 'Const':
        _raise_node_not_found()

    bn_param['moving_variance'] = tf.make_ndarray(const_node_def.attr['value'].tensor)
    return bn_param


def _extract_weights(model_pb_path):

    graph_def = tf.GraphDef()
    with tf.gfile.GFile(model_pb_path, 'rb') as file:
        file_data = file.read()
        graph_def.ParseFromString(file_data)

    graph_map = GraphMap(graph_def)

    with _LAYER_TO_NODE_PATH.open('r') as file:
        layer_to_node = json.load(file)

    layer_data = dict()
    _EPSILON = graph_map.get_node_by_name('mnasnet_1/lead_cell_12/op_0/bn2_0/FusedBatchNorm').attr['epsilon'].f
    for layer_name in layer_to_node:
        if layer_to_node[layer_name][0] is None:
            continue

        if len(layer_to_node[layer_name]) == 1:
            layer_data[layer_name] = get_layer_param(layer_to_node[layer_name][0], graph_map)

        if len(layer_to_node[layer_name]) == 2:
            layer_data[layer_name] = dict()
            bn = get_batch_norm_param(layer_to_node[layer_name][1], graph_map)
            w = get_layer_param(layer_to_node[layer_name][0], graph_map)['weights']

            w_scale = bn['gamma'] / (_EPSILON + bn['moving_variance']) ** (1 / 2)

            if w.shape[-1] == 1:
                w2 = w * np.expand_dims(w_scale, -1)
            else:
                w2 = w * w_scale

            if 'bias' in get_layer_param(layer_to_node[layer_name][0], graph_map):
                bias = get_layer_param(layer_to_node[layer_name][0], graph_map)['bias']
                bias2 = (bias - bn['moving_mean']) * w_scale + bn['beta']
            else:
                bias2 = -bn['moving_mean'] * w_scale + bn['beta']

            layer_data[layer_name]['weights'] = w2
            layer_data[layer_name]['bias'] = bias2

    return layer_data


def _save_data(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


@click.command()
@click.option('--pb-path', help='The original model', required=True, type=str)
def main(pb_path):
    # Step one: extract weights
    weights = _extract_weights(pb_path)
    _save_data(weights, 'model-data/weights_original.pickle')


if __name__ == '__main__':
    main()
