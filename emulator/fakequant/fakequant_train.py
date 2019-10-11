__all__ = [
    'train_adjustable_model',
    'LearningStrategy',
]

import logging
from collections import namedtuple
from pathlib import Path
from typing import List
from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow.contrib.graph_editor as ge

from emulator.dataset.base import BaseBatchStream
from emulator.fakequant.fakequant_adjustable_model import AdjustableThresholdsModel
from emulator.regular import RegularModel

LearningStrategy = namedtuple(
    'LearningStrategy',
    [
        'optimizer_type',
        'initial_lr',
        'lr_decay',
        'lr_cos_steps',
        'lr_cos_phase',
        'metric_type',
    ],
)

TrainOperations = namedtuple(
    'TrainOperations',
    [
        'gradients',
        'learning_rate',
        'loss',
    ],
)


_RMSE_EPS = 10 ** -5
_MINIMAL_LEARNING_RATE = 10 ** -7
_NUMBER_OF_REGULAR_CHECKPOINTS = 10
_NUMBER_OF_BEST_LOSS_CHECKPOINTS = 5
_REGULAR_CHECKPOINTS_DELTA = 50
_DEFAULT_LEARNING_STRATEGY = LearningStrategy(
    optimizer_type='adam',
    initial_lr=3e-4,
    lr_decay=8e-5,
    lr_cos_steps=500,
    lr_cos_phase=3.141592654 * 0.4,
    metric_type='rmse',
)
_DEFAULT_NUMBER_OF_EPOCHS = 10
_DEFAULT_MOMENTUM = 0.9

_LOGGER = logging.getLogger('emulator.train')


def train_adjustable_model(
        adjustable_network: AdjustableThresholdsModel,
        reference_network: RegularModel,
        batch_stream: BaseBatchStream,
        tensorboard_log_dir: Path, *,
        number_of_epochs: int = _DEFAULT_NUMBER_OF_EPOCHS,
        learning_strategy: LearningStrategy = _DEFAULT_LEARNING_STRATEGY,
) -> Path:

    trainable_graph, train_input, train_operations = _create_trainable_graph(
        adjustable_network, reference_network, learning_strategy,
    )

    with tf.Session(graph=trainable_graph) as session:
        session.run([
            var.initializer
            for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        ])
        best_loss_checkpoint_path = _train_adjustable_model(
            session=session,
            train_input=train_input,
            train_operations=train_operations,
            batch_stream=batch_stream,
            tensorboard_log_dir=tensorboard_log_dir,
            number_of_epochs=number_of_epochs,
        )
        adjustable_network.variables_checkpoint_path = best_loss_checkpoint_path

        return best_loss_checkpoint_path


def _copy_graph(src_graph: tf.Graph, dst_graph: tf.Graph, dst_scope: str = None):
    src_meta_graph = tf.train.export_meta_graph(graph=src_graph)
    with dst_graph.as_default():
        tf.train.import_meta_graph(src_meta_graph, import_scope=dst_scope)


def _get_transformed_tensor(src_tensor: tf.Tensor, dst_graph: tf.Graph, dst_scope: str = '') -> tf.Tensor:
    dst_tensor_name = src_tensor.name
    if dst_scope:
        dst_tensor_name = f'{dst_scope}/{dst_tensor_name}'

    return dst_graph.get_tensor_by_name(dst_tensor_name)


def _rmse_loss(reference_output, adjustable_output):
    return tf.reduce_mean(
        tf.reduce_sum(
            tf.sqrt(
                tf.subtract(adjustable_output, reference_output) ** 2 + _RMSE_EPS
            ),
            axis=1,
        )
    )


def _create_trainable_graph(
        adjustable_network: AdjustableThresholdsModel,
        reference_network: RegularModel,
        learning_strategy: LearningStrategy,
) -> Tuple[tf.Graph, tf.Tensor, TrainOperations]:

    adj_graph, [adj_input], [adj_output] = adjustable_network.graph_info
    ref_graph, [ref_input], [ref_output] = reference_network.graph_info

    trainable_graph = tf.Graph()
    _copy_graph(adj_graph, trainable_graph)
    _copy_graph(ref_graph, trainable_graph, 'reference_graph')

    adj_output = _get_transformed_tensor(adj_output, trainable_graph)
    ref_output = _get_transformed_tensor(ref_output, trainable_graph, 'reference_graph')
    adj_input = _get_transformed_tensor(adj_input, trainable_graph)
    ref_input = _get_transformed_tensor(ref_input, trainable_graph, 'reference_graph')

    with trainable_graph.as_default():  # pylint: disable=not-context-manager

        loss = _rmse_loss(ref_output, adj_output)

        learning_rate = tf.placeholder_with_default(
            learning_strategy.initial_lr,
            shape=[],
            name='lr_for_range_scalers',
        )

        optimizer_type = learning_strategy.optimizer_type
        optimizer_type = optimizer_type.lower()
        if optimizer_type == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate)
        else:
            raise NotImplementedError(f'optimizer "{optimizer_type}" is not supported')

        gradients = optimizer.compute_gradients(loss, var_list=tf.trainable_variables())
        gradients = optimizer.apply_gradients(gradients)

        train_operations = TrainOperations(gradients, learning_rate, loss)
        train_input = adj_input
        ge.reroute_ts([train_input], [ref_input])

        return trainable_graph, train_input, train_operations


def _train_adjustable_model(
        session: tf.Session,
        train_input: tf.Tensor,
        train_operations: TrainOperations,
        batch_stream: BaseBatchStream,
        tensorboard_log_dir: Path,
        number_of_epochs: int,
) -> Path:

    learning_strategy = _DEFAULT_LEARNING_STRATEGY

    thresholds_vars: List[tf.Variable] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    thresholds_vars_names = [th_var.name for th_var in thresholds_vars]
    global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    optimizer_vars = list(
        set(global_vars) - set(thresholds_vars)
    )

    _LOGGER.info(f'Total number of adjustable parameters {len(thresholds_vars)}')

    regular_checkpoints_saver = tf.train.Saver(
        var_list=thresholds_vars,
        max_to_keep=_NUMBER_OF_REGULAR_CHECKPOINTS,
    )
    best_loss_checkpoints_saver = tf.train.Saver(
        var_list=thresholds_vars,
        max_to_keep=_NUMBER_OF_BEST_LOSS_CHECKPOINTS,
    )
    summary_writer = tf.summary.FileWriter(str(tensorboard_log_dir), session.graph)

    loss_summary = tf.summary.scalar('train_loss', train_operations.loss)
    learning_rate_summary = tf.summary.scalar('train_learning_rate', train_operations.learning_rate)

    initial_learning_rate = session.run(train_operations.learning_rate)
    next_learning_rate = initial_learning_rate

    train_step = 0
    decay_step = 0
    nans_step_counter = 0

    best_loss = float('inf')
    best_loss_step = 0

    def reset_optimizer():
        init_optimizer_vars = tf.variables_initializer(optimizer_vars)
        session.run(init_optimizer_vars)

    ckpt_dir = tensorboard_log_dir / 'ckpt'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def get_regular_checkpoint_path(step):
        checkpoint_path = ckpt_dir / f'regular_ckpt_{step}'
        return str(checkpoint_path)

    def get_best_checkpoint_path(step):
        checkpoint_path = ckpt_dir / f'best_ckpt_{step}'
        return str(checkpoint_path)

    for epoch in range(number_of_epochs):
        _LOGGER.info(f'Epoch: {epoch + 1}')
        for batch_data, _ in batch_stream:
            feed_dict = {
                train_input: batch_data,
                train_operations.learning_rate: next_learning_rate,
            }
            fetches = {
                'gradients': train_operations.gradients,
                'loss': train_operations.loss,
                'learning_rate_summary': learning_rate_summary,
                'loss_summary': loss_summary,
            }

            result = session.run(fetches=fetches, feed_dict=feed_dict)

            next_learning_rate = initial_learning_rate.copy()
            next_learning_rate *= np.exp(-train_step * learning_strategy.lr_decay)
            next_learning_rate *= np.abs(
                np.cos(learning_strategy.lr_cos_phase * decay_step / learning_strategy.lr_cos_steps)
            )
            next_learning_rate += _MINIMAL_LEARNING_RATE

            summary_writer.add_summary(result['learning_rate_summary'], train_step)
            summary_writer.add_summary(result['loss_summary'], train_step)

            thresholds_vars_values = session.run(thresholds_vars)
            thresholds_with_nan = [
                np.isnan(var_value).any()
                for var_value in thresholds_vars_values
            ]
            thresholds_with_nan = [
                var_name
                for var_name, var_has_nan in zip(thresholds_vars_names, thresholds_with_nan)
                if var_has_nan
            ]

            if thresholds_with_nan:
                _LOGGER.warning(
                    f'Some thresholds are None, restore previous trainable values\n'
                    f'{thresholds_with_nan}'
                )

                nans_step_counter += 1
                checkpoint_step = train_step//_REGULAR_CHECKPOINTS_DELTA
                checkpoint_step -= nans_step_counter
                checkpoint_step = max(checkpoint_step, 0)
                checkpoint_step *= _REGULAR_CHECKPOINTS_DELTA

                regular_checkpoints_saver.restore(
                    session, get_regular_checkpoint_path(checkpoint_step)
                )
                reset_optimizer()
                continue

            if best_loss > result['loss']:
                best_loss_checkpoints_saver.save(session, get_best_checkpoint_path(train_step))
                best_loss, best_loss_step = result['loss'], train_step

            if train_step % _REGULAR_CHECKPOINTS_DELTA == 0:
                nans_step_counter = 0
                regular_checkpoints_saver.save(session, get_regular_checkpoint_path(train_step))

            train_step += 1
            decay_step += 1

            if train_step % learning_strategy.lr_cos_steps == 0:
                _LOGGER.info('Reinitialize optimizer')
                reset_optimizer()
                decay_step = 0

        _LOGGER.info(f'minimal loss value = {best_loss}')

    best_loss_checkpoints_saver.restore(
        session, get_best_checkpoint_path(best_loss_step)
    )
    best_checkpoint_path = get_best_checkpoint_path(best_loss_step)

    return Path(best_checkpoint_path)
