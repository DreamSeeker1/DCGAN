"""Discriminator used to discriminate real pics from generated pics"""
import tensorflow as tf
import params


def discriminator(input_pics):
    """Discriminator
    Args:
        input_pics: the input pics, assume [batch_size, 64, 64, 3]
    Returns:
        logits: the logits of the model
    """
    with tf.variable_scope('Discriminator'):
        # normalize the input by dividing 255.
        input_pics = input_pics / 255.
        # convolution layers
        conv1_layer = tf.layers.conv2d(input_pics, filters=128, activation=params.dis_conv_activation, kernel_size=5,
                                       strides=2,
                                       bias_initializer=tf.truncated_normal_initializer,
                                       padding='same', name='conv1')
        conv2_layer = tf.layers.conv2d(conv1_layer, filters=256, activation=params.dis_conv_activation, kernel_size=5,
                                       strides=2,
                                       bias_initializer=tf.truncated_normal_initializer,
                                       padding='same', name='conv2')
        conv3_layer = tf.layers.conv2d(conv2_layer, filters=512, activation=params.dis_conv_activation, kernel_size=5,
                                       strides=2,
                                       bias_initializer=tf.truncated_normal_initializer,
                                       padding='same', name='conv3')
        conv4_layer = tf.layers.conv2d(conv3_layer, filters=1024, activation=params.dis_conv_activation, kernel_size=5,
                                       strides=2,
                                       bias_initializer=tf.truncated_normal_initializer,
                                       padding='same', name='conv4')

        # reshape and project
        dense_input = tf.reshape(conv4_layer, shape=(-1, 4 * 4 * 1024))
        dense1_layer = tf.layers.dense(dense_input, 100, activation=params.dis_dense_activation,
                                       bias_initializer=tf.truncated_normal_initializer, name='dense1')
        # prediction
        dense2_layer = tf.layers.dense(dense1_layer, 2, name='dense2')
    return dense2_layer
