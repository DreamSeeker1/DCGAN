import tensorflow as tf

# activation function used in generator's convolution transpose layer
gen_activation = tf.nn.relu
# activation function used in discriminator
dis_conv_activation = tf.nn.relu
dis_dense_activation = tf.nn.relu

# optimizer used in  model
opt = tf.train.AdamOptimizer

# learning rate
lr = 0.001
