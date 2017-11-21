"""Discriminator used to discriminate real pics from generated pics"""
import tensorflow as tf
import model.params as params


def discriminator(input_pics, reuse, drop_prob):
    """Discriminator
    Args:
        input_pics: the input pics, assume [batch_size, 64, 64, 3]
        reuse: whether reuse the variables
        drop_prob: dropout probability
    Returns:
        logits: the logits of the model
    """
    with tf.variable_scope('Discriminator', reuse=reuse):
        # use the batch norm
        input_pics = tf.contrib.layers.batch_norm(input_pics)
        # convolution layers
        conv1_layer = tf.layers.conv2d(input_pics, filters=64, activation=params.dis_conv_activation, kernel_size=5,
                                       strides=2,
                                       bias_initializer=params.dis_bias_initializer,
                                       padding='same', name='conv1')
        conv2_layer = tf.layers.conv2d(conv1_layer, filters=128, activation=params.dis_conv_activation, kernel_size=5,
                                       strides=2,
                                       bias_initializer=params.dis_bias_initializer,
                                       padding='same', name='conv2')
        conv3_layer = tf.layers.conv2d(conv2_layer, filters=256, activation=params.dis_conv_activation, kernel_size=5,
                                       strides=2,
                                       bias_initializer=params.dis_bias_initializer,
                                       padding='same', name='conv3')
        conv4_layer = tf.layers.conv2d(conv3_layer, filters=512, activation=params.dis_conv_activation, kernel_size=5,
                                       strides=2,
                                       bias_initializer=params.dis_bias_initializer,
                                       padding='same', name='conv4')

        # reshape and project
        dense_input = tf.reshape(conv4_layer, shape=(-1, 4 * 4 * 512))
        dense_input_drop = tf.nn.dropout(dense_input, keep_prob=1 - drop_prob)
        output_layer = tf.layers.dense(dense_input_drop, 1, activation=params.dis_dense_activation,
                                       bias_initializer=params.dis_bias_initializer, name='output')
    return output_layer
