import numpy as np
import os.path
import shutil
import tempfile
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '4'
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.contrib.distribute import OneDeviceStrategy, MirroredStrategy


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    gpus = [dev.name for dev in local_device_protos if dev.device_type == 'GPU']
    return gpus


def model_fn(features, labels, mode, params):
    with tf.variable_scope('model'):
        # Build model
        x = tf.identity(features, name='model_input')
        for i in range(10):
            with tf.variable_scope('block_{}'.format(i)):
                x = tf.layers.conv2d(x, 16, (3, 3), (1, 1), padding='same', name='conv_1')
                x = tf.layers.conv2d(x, 16, (3, 3), (1, 1), padding='same', name='conv_2')
        with tf.variable_scope('down_0'):
            x = tf.layers.conv2d(x, 16, (3, 3), (2, 2), padding='same', name='conv_1')
            x = tf.layers.conv2d(x, 8, (3, 3), (2, 2), padding='same', name='conv_2')
        with tf.variable_scope('down_1'):
            x = tf.layers.conv2d(x, 4, (3, 3), (2, 2), padding='same', name='conv_3')
            x = tf.layers.conv2d(x, 2, (3, 3), (2, 2), padding='same', name='conv_4')
            x = tf.layers.conv2d(x, 1, (3, 3), (2, 2), padding='same', name='conv_5')

        x = tf.squeeze(x, name='predict')

    with tf.variable_scope('loss'):
        ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=x)
        loss = tf.reduce_mean(ce)

    with tf.variable_scope('train'):
        train_op = tf.train.AdamOptimizer().minimize(loss, global_step=tf.train.get_global_step())
        train_op = tf.group(tf.get_collection(tf.GraphKeys.UPDATE_OPS), train_op)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=x,
        loss=loss,
        train_op=train_op,
    )


def input_fn():
    def generator():
        for i in range(8):
            yield np.random.uniform(0, 1, (32, 32, 3)), 1.0

    ds = tf.data.Dataset.from_generator(
        generator=generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=((32, 32, 3), None)
    )
    ds = ds.repeat(2000)
    ds = ds.batch(8)
    return ds


def main(distribution_strategy_name, distribution_strategy):
    print('Running example for distribution_strategy {}'.format(distribution_strategy_name))
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)

    model_dir = os.path.join(tempfile.gettempdir(), 'read_x_example', distribution_strategy_name)
    print('writing logs to model dir {}'.format(model_dir))
    if os.path.exists(model_dir):
        print('removing existing model dir')
        shutil.rmtree(model_dir)

    run_config = tf.estimator.RunConfig(
        model_dir=model_dir,
        session_config=session_config,
        train_distribute=distribution_strategy,
        save_summary_steps=1,
        save_checkpoints_steps=1e4,
    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={},
    )

    estimator.train(input_fn, steps=1)

    print('Done with example {}'.format(distribution_strategy_name))
    print('run "tensorboard --logdir {}" to see the graph'.format(model_dir))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)
    available_gpus = get_available_gpus()
    main('MirroredStrategy2GPUs', MirroredStrategy(available_gpus[:2]))
    main('OneDeviceStrategy', OneDeviceStrategy(available_gpus[0]))
