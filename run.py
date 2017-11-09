"""the code that train the model"""
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import model.gen
import model.dis
import model.params as params

# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

test_gen_batch = np.ones(shape=(50, 4 * 4 * 1024))
test_real_batch = np.zeros(shape=(50, 64, 64, 3))


def get_labels(gen_pics, real_pics):
    """Generate labels of input pics
    Args:
        gen_pics: generated pics
        real_pics: real pics
    Returns:
        labels: a tensor of 0 and 1, 0 for generated, 1 for real pics
    """
    gen_pics_label = tf.tile(tf.constant([[0., 1.]], dtype=tf.float32), [tf.shape(gen_pics)[0], 1])
    real_pics_label = tf.tile(tf.constant([[1., 0.]], dtype=tf.float32), [tf.shape(real_pics)[0], 1])
    labels = tf.concat(
        [gen_pics_label, real_pics_label], axis=0)
    return labels


# define the computation graph
graph = tf.Graph()
with graph.as_default():
    # define the place holder for the input.
    gen_input = tf.placeholder(dtype=tf.float32, shape=(None, 4 * 4 * 1024))
    dis_input = tf.placeholder(dtype=tf.float32, shape=(None, 64, 64, 3))
    # generate pictures
    gen_pics = model.gen.generator(gen_input)
    # add labels to real and fake data.
    label = get_labels(gen_pics, dis_input)
    # go through the discriminator
    dis_logits = model.dis.discriminator(tf.concat([gen_pics, dis_input], axis=0))
    # compute loss
    discriminator_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=dis_logits))
    generator_loss = -discriminator_loss

    # define the optimizer used to train the discriminator
    dis_vars = graph.get_collection(name=tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')
    dis_opt = params.opt(params.lr)
    dis_grads_and_vars = dis_opt.compute_gradients(discriminator_loss, var_list=dis_vars)
    dis_capped_grads_and_vars = [(tf.clip_by_norm(gv[0], clip_norm=5.), gv[1]) for gv in dis_grads_and_vars]
    dis_opt_op = dis_opt.apply_gradients(dis_capped_grads_and_vars)

    # define the optimizer used to train the generator
    gen_vars = graph.get_collection(name=tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
    gen_opt = params.opt(params.lr)
    gen_grads_and_vars = dis_opt.compute_gradients(generator_loss, var_list=gen_vars)
    gen_capped_grads_and_vars = [(tf.clip_by_norm(gv[0], clip_norm=5.), gv[1]) for gv in gen_grads_and_vars]
    gen_opt_op = gen_opt.apply_gradients(dis_capped_grads_and_vars)

with tf.Session(graph=graph) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for step in range(50):
        for t in range(10):
            _, loss = sess.run([dis_opt_op, discriminator_loss],
                               feed_dict={gen_input: test_gen_batch, dis_input: test_real_batch})
            print(loss)
        _, loss = sess.run([gen_opt_op, generator_loss],
                           feed_dict={gen_input: test_gen_batch, dis_input: test_real_batch})
        print(loss)
