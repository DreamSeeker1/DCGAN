"""generator used to generate pictures"""

import tensorflow as tf
import params


def generator(input_batch):
    """Generator
    Args:
         input_batch: the input batch, assume the size is [batch_size, 100]
    Returns:
        pics_batch: the picture generator generates
    """
    with tf.variable_scope('Generator'):
        # project and reshape
        layer1 = tf.layers.dense(input_batch, units=4 * 4 * 1024)
        reshape_input = tf.reshape(layer1, shape=(-1, 4, 4, 1024))

        # convolution transpose
        conv1_trans = tf.layers.conv2d_transpose(reshape_input, filters=512, kernel_size=5, strides=(2, 2),
                                                 padding='same',
                                                 bias_initializer=tf.truncated_normal_initializer,
                                                 activation=params.gen_activation, name='conv1_trans')
        conv2_trans = tf.layers.conv2d_transpose(conv1_trans, filters=256, kernel_size=5, strides=(2, 2),
                                                 padding='same',
                                                 bias_initializer=tf.truncated_normal_initializer,
                                                 activation=params.gen_activation, name='conv2_trans')
        conv3_trans = tf.layers.conv2d_transpose(conv2_trans, filters=128, kernel_size=5, strides=(2, 2),
                                                 padding='same',
                                                 bias_initializer=tf.truncated_normal_initializer,
                                                 activation=params.gen_activation, name='conv3_trans')
        pics_batch = tf.layers.conv2d_transpose(conv3_trans, filters=3, kernel_size=5, strides=(2, 2),
                                                padding='same',
                                                bias_initializer=tf.truncated_normal_initializer,
                                                activation=params.gen_activation, name='conv4_trans')
        return pics_batch
