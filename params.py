import tensorflow as tf

# learning rate
lr = 0.0002
batch_size = 64
epoch = 10000
display_step = 10
max_model_number = 5
output_folder = './output'
# 1 for training the model, 2 for generate pictures
isTrain = 1
# train discriminator for k times in a step
k = 1

# optimizer used during training
opt = tf.train.AdamOptimizer
beta1 = 0.5
