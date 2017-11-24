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
    gen_pics_label = tf.zeros(shape=[tf.shape(gen_pics)[0], 1], dtype=tf.float32)
    real_pics_label = tf.ones(shape=[tf.shape(real_pics)[0], 1], dtype=tf.float32)
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
    gen_input = tf.placeholder(dtype=tf.float32, shape=(None, 100), name="noise")
    pics_input = tf.placeholder(dtype=tf.float32, shape=(None, 64, 64, mparams.channel), name="real_pics")
    drop_gen = tf.placeholder(dtype=tf.float32, shape=(), name='drop_prob_gen')
    drop_dis = tf.placeholder(dtype=tf.float32, shape=(), name='drop_prob_dis')
    # generate pictures
    gen_pics = model.gen.generator(gen_input, drop_gen)
    # add labels to real and fake data.
    label = get_labels(gen_pics, pics_input)
    # go through the discriminator
    dis_logits_fake = model.dis.discriminator(gen_pics, reuse=None, drop_prob=drop_dis)
    dis_logits_real = model.dis.discriminator(pics_input, reuse=True, drop_prob=drop_dis)
    dis_logits = tf.concat([dis_logits_fake, dis_logits_real], axis=0)
    # compute discriminator loss
    discriminator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=dis_logits))
    # compute generator loss
    # in this part we mark the label of fake pics as true, go through the discriminator and calculate loss
    fake_labels = tf.ones(shape=(tf.shape(gen_pics)[0], 1), dtype=tf.float32)
    generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=fake_labels,
                                                                            logits=dis_logits_fake))

    # define the saver to save the variables
    saver = tf.train.Saver(max_to_keep=params.max_model_number)

    with tf.variable_scope('train_discriminator'):
        # define the optimizer used to train the discriminator
        dis_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')
        dis_opt = params.opt(params.lr, beta1=params.beta1)
        dis_grads_and_vars = dis_opt.compute_gradients(discriminator_loss, var_list=dis_vars)
        dis_opt_op = dis_opt.apply_gradients(dis_grads_and_vars)
    with tf.variable_scope('train_generator'):
        # define the optimizer used to train the generator
        gen_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
        gen_opt = params.opt(params.lr, beta1=params.beta1)
        gen_grads_and_vars = dis_opt.compute_gradients(generator_loss, var_list=gen_vars)
        gen_opt_op = gen_opt.apply_gradients(gen_grads_and_vars)

    # the prediction accuracy of the discriminator, when equals to 50%, it can't distinguish
    # real and fake
    prediction = tf.cast(tf.greater(tf.sigmoid(dis_logits), tf.tile([0.5], [tf.shape(dis_logits)[0]])),
                         dtype=tf.float32)
    prediction_result = tf.equal(prediction, label)
    error_rate = 1 - tf.reduce_mean(tf.cast(prediction_result, tf.float32))

    # add summary
    tf.summary.scalar('Generator_loss', generator_loss)
    tf.summary.scalar('Discriminator_loss', discriminator_loss)
    tf.summary.scalar('Discriminator_error_rate', error_rate)
    merged = tf.summary.merge_all()

# define the session to run the training process
with tf.Session(graph=graph) as sess:
    if params.isTrain:
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
                for i in range(params.k):
                    # train the discriminator
                    try:
                        pics_in = sess.run(next_element)
                        # train the generator, use 100 dimensional uniform distribution
                        gen_in = np.random.uniform(-1., 1., size=(params.batch_size, 100))
                        _ = sess.run([dis_opt_op],
                                     {gen_input: gen_in, pics_input: pics_in, drop_gen: params.dropout_prob_gen,
                                      drop_dis: params.dropout_prob_dis})
                    except tf.errors.OutOfRangeError:
                        epoch_end = True
                        break
                if epoch_end:
                    break
                # train the generator, use 100 dimensional uniform distribution
                gen_in = np.random.uniform(-1., 1., size=(params.batch_size, 100))
                _ = sess.run(gen_opt_op, {gen_input: gen_in, pics_input: pics_in,
                                          drop_gen: params.dropout_prob_gen,
                                          drop_dis: params.dropout_prob_dis
                                          })
                summary = sess.run(merged, {gen_input: gen_in, pics_input: pics_in,
                                            drop_gen: params.dropout_prob_gen,
                                            drop_dis: 0.
                                            })
                step += 1
                # log summary
                summary_writer.add_summary(summary, step)

                if step % params.display_step == 0:
                    gen_loss, dis_loss, pics, err_rate = sess.run(
                        [generator_loss, discriminator_loss, gen_pics, error_rate],
                        {gen_input: gen_in, pics_input: pics_in, drop_gen: params.dropout_prob_gen,
                         drop_dis: 0.})
                    pic = ((pics[0] + 1.) / 2. * 255.).astype('uint8')
                    if mparams.channel == 3:
                        img = Image.fromarray(pic, 'RGB')
                    else:
                        pic.resize(64, 64)
                        img = Image.fromarray(pic, 'L')
                    img.save(os.path.join(params.output_folder, 'Epoch-{}-Step-{}.jpg'.format(epoch, step)))
                    print(
                        'Epoch: {}, Step: {}, Generator Loss: {:.4f}, Discriminator Loss:{:.4f}, Error Rate: {:.2%}'.format(
                            epoch, step,
                            gen_loss,
                            dis_loss,
                            err_rate
                        ))
            saver.save(sess, save_path='./checkpoint/model.ckpt', global_step=step)
    else:
        gen_in = np.random.uniform(-1., 1., size=(params.batch_size, 100))
        latest_checkpoint = tf.train.latest_checkpoint('./checkpoint/')
        saver.restore(sess, latest_checkpoint)
        gen_pics = sess.run(gen_pics, {gen_input: gen_in, drop_gen: params.dropout_prob_gen})
        gen_pics = ((gen_pics + 1.) / 2. * 255.).astype('uint8')
        for i in range(len(gen_pics)):
            pic = gen_pics[i]
            if mparams.channel == 3:
                img = Image.fromarray(pic, 'RGB')
            else:
                pic.resize(64, 64)
                img = Image.fromarray(pic, 'L')
            img.save(os.path.join(params.output_folder, 'Generated-{}.jpg'.format(i + 1)))
