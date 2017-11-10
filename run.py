"""the code that train the model"""
import tensorflow as tf
import numpy as np
import model.gen
import model.dis
import model.params as mparams
import data_utils.tf_utils
from PIL import Image
import params
import os

# check if the output folder exists
if not os.path.exists(params.output_folder):
    os.mkdir(params.output_folder)


def get_labels(gen_pics, real_pics):
    """Generate true labels of input pics
    Args:
        gen_pics: generated pics
        real_pics: real pics
    Returns:
        labels: a tensor of 0 and 1, 0 for generated, 1 for real pics
    """
    gen_pics_label = tf.tile(tf.constant([[0, 1]], dtype=tf.int32), [tf.shape(gen_pics)[0], 1])
    real_pics_label = tf.tile(tf.constant([[1, 0]], dtype=tf.int32), [tf.shape(real_pics)[0], 1])
    labels = tf.concat(
        [gen_pics_label, real_pics_label], axis=0)
    return labels


# define the computation graph
graph = tf.Graph()
with graph.as_default():
    # load data
    dataset = data_utils.tf_utils.get_pics('./pics').batch(params.batch_size).shuffle(2 * params.batch_size)
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
    # compute discriminator loss
    discriminator_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=dis_logits))
    # compute generator loss
    # in this part we mark the label of fake pics as true, go through the discriminator and calculate loss
    fake_labels = tf.tile(tf.constant([[1, 0]], dtype=tf.int32), [tf.shape(gen_pics)[0], 1])
    generator_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=fake_labels,
                                                                            logits=dis_logits[:params.batch_size]))

    # define the saver to save the variables
    saver = tf.train.Saver(max_to_keep=params.max_model_number)

    with tf.name_scope('train_discriminator'):
        # define the optimizer used to train the discriminator
        dis_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')
        dis_opt = mparams.opt(params.lr)
        dis_grads_and_vars = dis_opt.compute_gradients(discriminator_loss, var_list=dis_vars)
        dis_capped_grads_and_vars = [(tf.clip_by_norm(gv[0], clip_norm=5.), gv[1]) for gv in dis_grads_and_vars]
        dis_opt_op = dis_opt.apply_gradients(dis_capped_grads_and_vars)

    with tf.name_scope('train_generator'):
        # define the optimizer used to train the generator
        gen_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
        gen_opt = mparams.opt(params.lr)
        gen_grads_and_vars = dis_opt.compute_gradients(generator_loss, var_list=gen_vars)
        gen_capped_grads_and_vars = [(tf.clip_by_norm(gv[0], clip_norm=5.), gv[1]) for gv in gen_grads_and_vars]
        gen_opt_op = gen_opt.apply_gradients(gen_capped_grads_and_vars)

    # the prediction result of the discriminator when input is the fake data
    # the higher the better
    false_prediction = tf.equal(tf.argmax(dis_logits[:params.batch_size], 1, output_type=tf.int32),
                                tf.argmax(fake_labels, 1, output_type=tf.int32))
    error_rate = tf.reduce_mean(tf.cast(false_prediction, tf.float32))
    # add summary
    tf.summary.scalar('Generator_loss', generator_loss)
    tf.summary.scalar('Discriminator_loss', discriminator_loss)
    tf.summary.scalar('Discriminator_error_rate', error_rate)
    merged = tf.summary.merge_all()

# define the session to run the training process
with tf.Session(graph=graph) as sess:
    summary_writer = tf.summary.FileWriter('./Graph', sess.graph)
    # define step
    step = 0
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(params.epoch):
        sess.run(iterator.initializer)
        epoch_end = False
        while True:
            pics_in = None
            gen_in = None
            for i in range(5):
                # train the discriminator
                try:
                    pics_in = sess.run(next_element)
                    gen_in = np.random.rand(params.batch_size, 4 * 4 * 1024)
                    _ = sess.run([dis_opt_op], {gen_input: gen_in, pics_input: pics_in})
                except tf.errors.OutOfRangeError:
                    epoch_end = True
                    break
            if epoch_end:
                break
            # train the generator
            gen_in = np.random.rand(params.batch_size, 4 * 4 * 1024)
            _ = sess.run([gen_opt_op], {gen_input: gen_in, pics_input: pics_in})
            step += 1
            # log summary
            summary = sess.run(merged, {gen_input: gen_in, pics_input: pics_in})
            summary_writer.add_summary(summary, step)
            saver.save(sess, save_path='./checkpoint/model.ckpt', global_step=step)

            if step % params.display_step == 0:
                gen_loss, dis_loss, pics, err_rate = sess.run(
                    [generator_loss, discriminator_loss, gen_pics, error_rate],
                    {gen_input: gen_in, pics_input: pics_in})
                pic = pics[0]
                img = Image.fromarray(pic, 'RGB')
                img.save(os.path.join(params.output_folder, 'Epoch-{}-Step-{}.jpg'.format(epoch, step)))
                print(
                    'Epoch: {}, Step: {}, Generator Loss: {:.4f}, Discriminator Loss:{:.4f}, Error Rate: {:.2%}'.format(
                        epoch, step,
                        gen_loss,
                        dis_loss,
                        err_rate
                    ))
