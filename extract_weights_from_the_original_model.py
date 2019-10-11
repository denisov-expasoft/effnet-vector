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


def create_graphdef_from_ckpt(ckpt_dir):
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        new_saver = tf.train.import_meta_graph(ckpt_dir + 'model.ckpt.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))

        graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph_def,
            ['logits'],
        )

    node_to_delete = []
    for i, node in enumerate(graph_def.node):
        if node.name in ('truediv', 'sub', 'IteratorGetNext', 'OneShotIterator'):
            node_to_delete.append(i)
    node_to_delete.sort(reverse=True)
    for i in node_to_delete:
        del graph_def.node[i]
    graph = tf.Graph()
    with graph.as_default():
        _ = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='truediv')
        placeholder_node_def = graph.as_graph_def().node[0]

    graph_def.node.extend([placeholder_node_def])

    return graph_def


def get_batch_norm_param(layer_name, graph_map):
    # gamma
    bn_param = dict()
    node_def = graph_map.get_node_by_name(layer_name)
    mul_node_def_gamma = graph_map.get_node_by_name(node_def.input[1])
    if mul_node_def_gamma.op != 'Mul':
        _raise_node_not_found()

    identity_node_def = graph_map.get_node_by_name(mul_node_def_gamma.input[1])
    if identity_node_def.op != 'Identity':
        _raise_node_not_found()

    const_node_def = graph_map.get_node_by_name(identity_node_def.input[0])
    if const_node_def.op != "Const":
        _raise_node_not_found()

    bn_param['gamma'] = tf.make_ndarray(const_node_def.attr['value'].tensor)

    # beta
    node_def = graph_map.get_node_by_name(layer_name)
    add_node_def = graph_map.get_node_consumers_nodedef(node_def.name)[0]
    if add_node_def.op != 'AddV2':
        _raise_node_not_found()

    sub_node_def = graph_map.get_node_by_name(add_node_def.input[1])
    if sub_node_def.op != "Sub":
        _raise_node_not_found()

    identity_node_def = graph_map.get_node_by_name(sub_node_def.input[0])
    if identity_node_def.op != 'Identity':
        _raise_node_not_found()

    const_node_def = graph_map.get_node_by_name(identity_node_def.input[0])
    if const_node_def.op != "Const":
        _raise_node_not_found()

    bn_param['beta'] = tf.make_ndarray(const_node_def.attr['value'].tensor)

    # moving_mean
    mul_node_def_mean = graph_map.get_node_by_name(sub_node_def.input[1])

    if mul_node_def_mean.op != 'Mul':
        _raise_node_not_found()

    identity_node_def = graph_map.get_node_by_name(mul_node_def_mean.input[0])
    if identity_node_def.op != 'Identity':
        _raise_node_not_found()

    const_node_def = graph_map.get_node_by_name(identity_node_def.input[0])
    if const_node_def.op != "Const":
        _raise_node_not_found()

    bn_param['moving_mean'] = tf.make_ndarray(const_node_def.attr['value'].tensor)

    # moving_variance
    rsqrt_node_def = graph_map.get_node_by_name(mul_node_def_gamma.input[0])
    if rsqrt_node_def.op != 'Rsqrt':
        _raise_node_not_found()

    add_node_def = graph_map.get_node_by_name(rsqrt_node_def.input[0])
    if add_node_def.op != 'AddV2':
        _raise_node_not_found()

    identity_node_def = graph_map.get_node_by_name(add_node_def.input[0])
    if identity_node_def.op != 'Identity':
        _raise_node_not_found()

    const_node_def = graph_map.get_node_by_name(identity_node_def.input[0])
    if const_node_def.op != "Const":
        _raise_node_not_found()

    bn_param['moving_variance'] = tf.make_ndarray(const_node_def.attr['value'].tensor)

    return bn_param


def _extract_weights(ckpt_dir_path):

    graph_def = create_graphdef_from_ckpt(ckpt_dir_path)

    graph_map = GraphMap(graph_def)

    with _LAYER_TO_NODE_PATH.open('r') as file:
        layer_to_node = json.load(file)

    layer_data = dict()
    _EPSILON = graph_map.get_node_by_name(
        'efficientnet-b0/model/blocks_4/tpu_batch_normalization_2/batchnorm/add/y'
    ).attr['epsilon'].f
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
@click.option('--ckpt-dir-path', help='The original model', required=True, type=str)
def main(ckpt_dir_path):
    # Step one: extract weights
    weights = _extract_weights(ckpt_dir_path)
    _save_data(weights, 'model-data/weights.pickle')


if __name__ == '__main__':
    main()
