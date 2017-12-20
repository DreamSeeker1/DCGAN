"""generator used to generate pictures"""

import tensorflow as tf
import model.params as params
from model.params import bn


def generator(input_batch, drop_prob):
    """Generator
    Args:
         input_batch: the input batch, assume the size is [batch_size, 100]
         drop_prob: dropout probability
    Returns:
        pics_batch: the picture generator generates
    """
    # activation function for convolution layer and dense layer
    gen_act = params.gen_activation
    output_act = params.gen_output_activation

    with tf.variable_scope('Generator'):
        # define batch normalization layers
        bn0 = bn(name='bn0', renorm=True)
        bn1 = bn(name='bn1', renorm=True)
        bn2 = bn(name='bn2', renorm=True)
        bn3 = bn(name='bn3', renorm=True)

        # project and reshape
        layer1 = tf.layers.dense(input_batch, units=4 * 4 * 1024)
        reshape_input = tf.reshape(layer1, shape=(-1, 4, 4, 1024))
        # use batch normalization
        reshape_input = gen_act(bn0(reshape_input))

        # convolution transpose
        conv1_trans = tf.layers.conv2d_transpose(reshape_input, filters=512, kernel_size=5, strides=(2, 2),
                                                 padding='same',
                                                 bias_initializer=params.gen_bias_initializer,
                                                 name='conv1_trans')
        # use batch normalization
        conv1_trans = gen_act(bn1(conv1_trans))
        # use dropout
        conv1_trans_drop = tf.nn.dropout(conv1_trans, keep_prob=1 - drop_prob)
        conv2_trans = tf.layers.conv2d_transpose(conv1_trans_drop, filters=256, kernel_size=5, strides=(2, 2),
                                                 padding='same',
                                                 bias_initializer=params.gen_bias_initializer,
                                                 name='conv2_trans')
        conv2_trans = gen_act(bn2(conv2_trans))
        conv2_trans_drop = tf.nn.dropout(conv2_trans, keep_prob=1 - drop_prob)
        conv3_trans = tf.layers.conv2d_transpose(conv2_trans_drop, filters=128, kernel_size=5, strides=(2, 2),
                                                 padding='same',
                                                 bias_initializer=params.gen_bias_initializer,
                                                 name='conv3_trans')
        conv3_trans = gen_act(bn3(conv3_trans))
        conv3_trans_drop = tf.nn.dropout(conv3_trans, keep_prob=1 - drop_prob)
        # use Tanh in output layer
        pics_batch = tf.layers.conv2d_transpose(conv3_trans_drop, filters=params.channel, kernel_size=5, strides=(2, 2),
                                                padding='same',
                                                bias_initializer=params.gen_bias_initializer,
                                                name='conv4_trans')
        pics_batch = output_act(pics_batch)
        return pics_batch
