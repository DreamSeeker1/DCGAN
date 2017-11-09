"""the code that train the model"""
import tensorflow as tf
import numpy as np
import model.gen
import model.dis
import model.params as params
import data_utils.tf_utils
from PIL import Image


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
    # load data
    dataset = data_utils.tf_utils.get_pics('./pics').batch(params.batch_size)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    # define the place holder for the input.
    gen_input = tf.placeholder(dtype=tf.float32, shape=(None, 4 * 4 * 1024))
    pics_input = tf.placeholder(dtype=tf.float32, shape=(None, 64, 64, 3))
    # generate pictures
    gen_pics = model.gen.generator(gen_input)
    # add labels to real and fake data.
    label = get_labels(gen_pics, pics_input)
    # go through the discriminator
    dis_logits = model.dis.discriminator(tf.concat([gen_pics, pics_input], axis=0))
    # compute loss
    discriminator_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=dis_logits))
    generator_loss = -discriminator_loss

    # define the optimizer used to train the discriminator
    dis_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')
    dis_opt = params.opt(params.lr)
    dis_grads_and_vars = dis_opt.compute_gradients(discriminator_loss, var_list=dis_vars)
    dis_capped_grads_and_vars = [(tf.clip_by_norm(gv[0], clip_norm=5.), gv[1]) for gv in dis_grads_and_vars]
    dis_opt_op = dis_opt.apply_gradients(dis_capped_grads_and_vars)

    # define the optimizer used to train the generator
    gen_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
    gen_opt = params.opt(params.lr)
    gen_grads_and_vars = dis_opt.compute_gradients(generator_loss, var_list=gen_vars)
    gen_capped_grads_and_vars = [(tf.clip_by_norm(gv[0], clip_norm=5.), gv[1]) for gv in gen_grads_and_vars]
    gen_opt_op = gen_opt.apply_gradients(gen_capped_grads_and_vars)

# define the session to run the training process
with tf.Session(graph=graph) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(params.epoch):
        sess.run(iterator.initializer)
        while True:
            try:
                pics_in = sess.run(next_element)
                gen_in = np.random.rand(params.batch_size, 4 * 4 * 1024)
            except tf.errors.OutOfRangeError:
                break
            for i in range(5):
                _, loss = sess.run([dis_opt_op, discriminator_loss], {gen_input: gen_in, pics_input: pics_in})
                print('Discriminator Loss: {:.4f}'.format(loss))
            _, loss, pic = sess.run([gen_opt_op, generator_loss, gen_pics],
                                    {gen_input: gen_in, pics_input: pics_in})
            img = Image.fromarray(pic[0], 'RGB')
            img.save('%d.png' % epoch)
            print('Generator Loss {:.4f}'.format(loss))
            print('save')
