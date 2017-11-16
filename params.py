import tensorflow as tf

# learning rate
lr = 0.0002
batch_size = 256
epoch = 10000
display_step = 10
max_model_number = 15
output_folder = './output'
# 1 for training the model, 2 for generate pictures
isTrain = 0
# train discriminator for k times in a step
k = 1

# optimizer used during training
opt = tf.train.AdamOptimizer
