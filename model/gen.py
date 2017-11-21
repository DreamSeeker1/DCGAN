"""generator used to generate pictures"""

import tensorflow as tf
import model.params as params


def generator(input_batch, drop_prob):
    """Generator
    Args:
         input_batch: the input batch, assume the size is [batch_size, 100]
         drop_prob: dropout probability
    Returns:
        pics_batch: the picture generator generates
    """
    with tf.variable_scope('Generator'):
        # add batch normalization
        input_batch = tf.contrib.layers.batch_norm(input_batch)
        # project and reshape
        layer1 = tf.layers.dense(input_batch, units=4 * 4 * 1024)
        reshape_input = tf.reshape(layer1, shape=(-1, 4, 4, 1024))

        # convolution transpose
        conv1_trans = tf.layers.conv2d_transpose(reshape_input, filters=512, kernel_size=5, strides=(2, 2),
                                                 padding='same',
                                                 bias_initializer=params.gen_bias_initializer,
                                                 activation=params.gen_activation, name='conv1_trans')
        conv1_trans_drop = tf.nn.dropout(conv1_trans, keep_prob=1 - drop_prob)
        conv2_trans = tf.layers.conv2d_transpose(conv1_trans_drop, filters=256, kernel_size=5, strides=(2, 2),
                                                 padding='same',
                                                 bias_initializer=params.gen_bias_initializer,
                                                 activation=params.gen_activation, name='conv2_trans')
        conv2_trans_drop = tf.nn.dropout(conv2_trans, keep_prob=1 - drop_prob)
        conv3_trans = tf.layers.conv2d_transpose(conv2_trans_drop, filters=128, kernel_size=5, strides=(2, 2),
                                                 padding='same',
                                                 bias_initializer=params.gen_bias_initializer,
                                                 activation=params.gen_activation, name='conv3_trans')
        conv3_trans_drop = tf.nn.dropout(conv3_trans, keep_prob=1 - drop_prob)
        # use Tanh in output layer
        pics_batch = tf.layers.conv2d_transpose(conv3_trans_drop, filters=params.channel, kernel_size=5, strides=(2, 2),
                                                padding='same',
                                                bias_initializer=params.gen_bias_initializer,
                                                activation=params.gen_activation, name='conv4_trans')
        pics_batch = params.gen_output_activation(pics_batch)
        return pics_batch
