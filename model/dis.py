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
        # convolution layers
        conv1_layer = tf.layers.conv2d(input_pics, filters=128, activation=params.dis_conv_activation, kernel_size=5,
                                       strides=1,
                                       bias_initializer=tf.truncated_normal_initializer,
                                       padding='same', name='conv1')
        pool1_layer = tf.layers.max_pooling2d(conv1_layer, pool_size=2, strides=2, name='pool1')
        conv2_layer = tf.layers.conv2d(pool1_layer, filters=256, activation=params.dis_conv_activation, kernel_size=5,
                                       strides=1,
                                       bias_initializer=tf.truncated_normal_initializer,
                                       padding='same', name='conv2')
        pool2_layer = tf.layers.max_pooling2d(conv2_layer, pool_size=2, strides=2, name='pool2')
        conv3_layer = tf.layers.conv2d(pool2_layer, filters=512, activation=params.dis_conv_activation, kernel_size=5,
                                       strides=1,
                                       bias_initializer=tf.truncated_normal_initializer,
                                       padding='same', name='conv3')
        pool3_layer = tf.layers.max_pooling2d(conv3_layer, pool_size=2, strides=2, name='pool3')
        conv4_layer = tf.layers.conv2d(pool3_layer, filters=1024, activation=params.dis_conv_activation, kernel_size=5,
                                       strides=1,
                                       bias_initializer=tf.truncated_normal_initializer,
                                       padding='same', name='conv4')
        pool4_layer = tf.layers.max_pooling2d(conv4_layer, pool_size=2, strides=2, name='pool4')

        # reshape and project
        dense_input = tf.reshape(pool4_layer, shape=(-1, 4 * 4 * 1024))
        dense1_layer = tf.layers.dense(dense_input, 100, activation=params.dis_dense_activation,
                                       bias_initializer=tf.truncated_normal_initializer, name='dense1')
        # prediction
        dense2_layer = tf.layers.dense(dense1_layer, 2, name='dense2')
    return dense2_layer
