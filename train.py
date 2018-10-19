import tensorflow as tf
import numpy as np


g_depth = [1024, 512, 256, 128, 1]
g_length = [4, 8, 16, 32, 64]
d_depth = [64, 128, 256, 512]
d_length = []

nosie_dim = 100
batch_size = 100
learning_rate = 0.03

def generator(z):
    g_h0 = mat_operation(z, g_length[0]*g_length[0]*g_depth[0], 'gen_vars')
    g_h0 = tf.reshape(g_h0, [-1,g_length[0],g_length[0], g_depth[0]])
    
    g_h1 = conv2d_transpose(input=g_h0,
                            output_shape=[batch_size, g_length[1], g_length[1], g_depth[1]], 
                            name='gen_var')
    g_h1 = batch_normalization_and_relu(g_h1, "gen")

    g_h2 = conv2d_transpose(input=g_h1,
                            output_shape=[batch_size, g_length[2], g_length[2], g_depth[2]], 
                            name='gen_var')
    g_h2 = batch_normalization_and_relu(g_h1, "gen")

    g_h3 = conv2d_transpose(input=g_h2,
                            output_shape=[batch_size, g_length[3], g_length[3], g_depth[3]], 
                            name='gen_var')
    g_h3 = batch_normalization_and_relu(g_h3, "gen")

    g_h4 = conv2d_transpose(input=g_h3,
                            output_shape=[batch_size, g_length[4], g_length[4], g_depth[4]], 
                            name='gen_var')
    g_h4 = tf.nn.tanh(g_h4)

    return g_h4


def discriminator(x):

    d_h0 = conv2d(input=x,
                  output_depth=d_depth[1],
                  name='disc_vars')
    d_h0 = batch_normalization_and_relu(d_h0, "disc")

    d_h1 = conv2d(input=d_h0,
                  output_depth=d_depth[2],
                  name='disc_vars')
    d_h1 = batch_normalization_and_relu(d_h1, "disc")

    d_h2 = conv2d(input=d_h1,
                  output_depth=d_depth[3],
                  name='disc_vars')
    d_h2 = batch_normalization_and_relu(d_h2, "disc")

    d_h3 = conv2d(input=g_h2,
                  output_depth=d_depth[4],
                  name='disc_vars')

    output = mat_operation(d_h3, 1, name='disc_vars')
    output = tf.nn.sigmoid(output)

    return output


def conv2d_transpose(input, output_shape, name ,k_h=5, k_w=5):
    
    filter_shape = [k_h, k_w, output_shape[-1], input.get_shape()[-1]]

    with tf.variable_scope(name):

        W = tf.Variable(tf.random_normal(shape=filter_shape, stddev=5e-2))
        output = tf.nn.conv2d_transpose(input, W, output_shape = output_shape, strides=[1,k_h,k_w,1])

    return output


def conv2d(input, output_depth,name, k_h=5, k_w=5):

    filter_shape = [k_h, k_w, input.get_shape()[-1], output_depth]

    with tf.variable_scope(name):

        W = tf.Variable(tf.random_normal(shape=filter_shape, stddev=5e-2))
        b = tf.Variable(tf.constant(0.1, shape=[output_depth]))
        output = tf.nn.conv2d(input, W, strides=[1,k_h,k_w,1], padding="SAME")+b
        
    return output
    

def batch_normalization_and_relu(layer, name):
    if(name == "gen"):
        output = tf.nn.relu(tf.layers.batch_normalization(layer, training=training))
    elif(name == "disc"):
        output = tf.nn.leaky_relu(layer, alpha=0.2)

    return output


def mat_operation(input, output_size, name):
    shape = [input[-1], output_size]

    with tf.variable_scope(name):

        W = tf.variable(tf.random_normal(shape=shape), stddev=5e-2)
        b = tf.variable(tf.constant(0.1, shape=[output_size]))

    output = tf.matmul(input, W) + b

    return output

def main():
    # placeholder
    gen_input = tf.placeholder(tf.float32, [None, nosie_dim])
    disc_input = tf.placeholder(tf.float32, [None, 64,64,3])

    gen_output = generator(gen_input)

    fake_output = discriminator(gen_output)
    real_output = discriminator(disc_input)

    disc_loss = -tf.reduce_mean(tf.log(real_output) + tf.log(1. - fake_output))
    gen_loss = -tf.reduce_mean(tf.log(fake_output))

    tvar = tf.trainable_variables()
    gvar = [var for var in tvar if 'gen' in var.name]
    dvar = [var for var in tvar if 'disc' in var.name]

    gen_train_step = tf.train.AdamOptimizer(learning_rate).minimize(gen_loss, var_list=gvar)
    disc_train_step = tf.train.AdamOptimizer(learning_rate).minimize(disc_loss, var_list=dvar)

