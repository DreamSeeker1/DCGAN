"""Discriminator used to discriminate real pics from generated pics"""
import tensorflow as tf
import model.params as params
from model.params import bn as bn

# activation function for convolution layer and dense layer
dis_act = params.dis_activation
dens_act = params.dis_dense_activation


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
        # define the batch normalization layers
        bn0 = bn(name='bn0', renorm=True)
        bn1 = bn(name='bn1', renorm=True)
        bn2 = bn(name='bn2', renorm=True)
        bn3 = bn(name='bn3', renorm=True)

        # convolution layers and batch norm layers
        conv1_layer = tf.layers.conv2d(input_pics, filters=64, kernel_size=5,
                                       strides=2,
                                       bias_initializer=params.dis_bias_initializer,
                                       padding='same', name='conv1')
        conv1_layer = dis_act(bn0(conv1_layer))
        conv2_layer = tf.layers.conv2d(conv1_layer, filters=128, kernel_size=5,
                                       strides=2,
                                       bias_initializer=params.dis_bias_initializer,
                                       padding='same', name='conv2')
        conv2_layer = dis_act(bn1(conv2_layer))
        conv3_layer = tf.layers.conv2d(conv2_layer, filters=256, kernel_size=5,
                                       strides=2,
                                       bias_initializer=params.dis_bias_initializer,
                                       padding='same', name='conv3')
        conv3_layer = dis_act(bn2(conv3_layer))
        conv4_layer = tf.layers.conv2d(conv3_layer, filters=512, kernel_size=5,
                                       strides=2,
                                       bias_initializer=params.dis_bias_initializer,
                                       padding='same', name='conv4')
        conv4_layer = dis_act(bn3(conv4_layer))
        # reshape and project
        dense_input = tf.reshape(conv4_layer, shape=(-1, 4 * 4 * 512))
        dense_input_drop = tf.nn.dropout(dense_input, keep_prob=1 - drop_prob)
        # output logits
        output_layer = tf.layers.dense(dense_input_drop, 1,
                                       bias_initializer=params.dis_bias_initializer, name='output')
    return output_layer
