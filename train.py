import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from ops import *
   
g_depth = [512, 256, 128, 64, 3]
g_length = [4, 8, 16, 32, 64]
d_depth = [64, 128, 256, 512, 512]
d_length = []

img_width = 28
img_height = 28


nosie_dim = 100
batch_size = 100
learning_rate = 0.03
epoch = 1

mnist = input_data.read_data_sets("data/", one_hot=True)

def generator(Z):

    g_h0 = mat_operation(Z, g_length[0]*g_length[0]*g_depth[0], 'gen_vars')
    g_h0 = tf.reshape(g_h0, [-1,g_length[0],g_length[0], g_depth[0]])

    g_h1 = conv2d_transpose(input=g_h0,
                            output_shape=[batch_size, g_length[1], g_length[1], g_depth[1]], 
                            name='gen_var')
#    print(g_h1.get_shape().as_list())

    g_h2 = conv2d_transpose(input=g_h1,
                            output_shape=[batch_size, g_length[2], g_length[2], g_depth[2]], 
                            name='gen_var')
    g_h2 = batch_normalization_and_relu(g_h2, "gen")
    
#    print(g_h2.get_shape().as_list())

    g_h3 = conv2d_transpose(input=g_h2,
                            output_shape=[batch_size, g_length[3], g_length[3], g_depth[3]], 
                            name='gen_var')
    g_h3 = batch_normalization_and_relu(g_h3, "gen")
    
#    print(g_h3.get_shape().as_list())


    g_h4 = conv2d_transpose(input=g_h3,
                            output_shape=[batch_size, g_length[4], g_length[4], g_depth[4]], 
                            name='gen_var')
    g_h4 = tf.nn.tanh(g_h4)

    return g_h4


def discriminator(x):

    x = tf.reshape(x, [100, 64, 64, 3])

    d_h0 = conv2d(input=x,
                  output_depth=d_depth[0],
                  name='disc_vars')

    print(d_h0.get_shape().as_list())

    d_h1 = conv2d(input=d_h0,
                  output_depth=d_depth[1],
                  name='disc_vars')
    d_h1 = batch_normalization_and_relu(d_h1, "disc")

    print(d_h1.get_shape().as_list())

    d_h2 = conv2d(input=d_h1,
                  output_depth=d_depth[2],
                  name='disc_vars')
    d_h2 = batch_normalization_and_relu(d_h2, "disc")

    print(d_h2.get_shape().as_list())

    d_h3 = conv2d(input=d_h2,
                  output_depth=d_depth[3],
                  name='disc_vars')

    d_h3_shape = d_h3.get_shape().as_list()
    d_h3 = tf.reshape(d_h3, [batch_size, d_h3_shape[1]*d_h3_shape[2]*d_h3_shape[3]])

    print(d_h3.get_shape().as_list())

    output = mat_operation(d_h3, 1, name='disc_vars')
    output = tf.nn.sigmoid(output)

    return output


def train():
    # placeholder
    
    gen_input = tf.placeholder(tf.float32, [None, nosie_dim])
    disc_input = tf.placeholder(tf.float32, [None, img_width, img_height, 1])

    gen_output = generator(gen_input)

    print(gen_output.get_shape().as_list())
    fake_output = discriminator(gen_output)
    real_output = discriminator(disc_input)


    disc_loss = -tf.reduce_mean(tf.log(real_output) + tf.log(1. - fake_output))
    gen_loss = -tf.reduce_mean(tf.log(fake_output))


    tvar = tf.trainable_variables()
    gvar = [var for var in tvar if 'gen' in var.name]
    dvar = [var for var in tvar if 'disc' in var.name]

    gen_train_step = tf.train.AdamOptimizer(learning_rate).minimize(gen_loss, var_list=gvar)
    disc_train_step = tf.train.AdamOptimizer(learning_rate).minimize(disc_loss, var_list=dvar)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(epoch):
            train_real_batch, _ = mnist.train.next_batch(batch_size)

            print(np.shape(train_real_batch))


if __name__ == "__main__":
    train()
