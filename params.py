import tensorflow as tf

# learning rate
lr = 0.002
batch_size = 256
epoch = 10000
display_step = 10
max_model_number = 15
output_folder = './output'

# train discriminator for k times in a step
k = 2

# optimizer used during training
opt = tf.train.AdamOptimizer
