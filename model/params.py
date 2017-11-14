import tensorflow as tf

# activation function used in generator's convolution transpose layer
gen_activation = tf.nn.relu
gen_output_activation = tf.nn.tanh
# activation function used in discriminator
dis_conv_activation = tf.nn.leaky_relu
dis_dense_activation = tf.nn.leaky_relu

# optimizer used in  model
opt = tf.train.AdamOptimizer
