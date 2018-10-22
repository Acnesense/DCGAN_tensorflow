import tensorflow as tf
import numpy as np
import os
import sys
import cPickle
from ops import *
from utils import *

print("image loading start")

dir_path = "cifar-10"
data = []

if os.environ.get('DISPLAY', '')== '':
    plt.switch_backend('agg')


for i in range(1,6):
    file_name = "data_batch_" + str(i)
    file_path = os.path.join(dir_path, file_name)
    with open(file_path, 'rb') as file:
        image_dict = cPickle.load(file)

    image = image_dict["data"]
    for img in image:
        img = img.reshape(32,32,3)
        data.append(img)

data = np.array(data)
print(np.shape(data))
print("image is loaded")

img_width = 32
img_height = 32

g_depth = [512, 256, 128, 64, 3]
g_length = [2, 4, 8, 16, 32]
d_depth = [64, 128, 256, 512, 512]
d_length = []

rgb = True

if rgb is True:
    color = 3

data_length = len(data)

noise_dim = 100
batch_size = 200
learning_rate = 0.00005
epoch = 100

#mnist = input_data.read_data_sets("data/", one_hot=True)

def generator(Z):



#    print(Z.get_shape().as_list())
    g_h0 = mat_operation(Z, g_length[0]*g_length[0]*g_depth[0], 'gen_vars')
    g_h0 = tf.reshape(g_h0, [-1,g_length[0],g_length[0], g_depth[0]])

#    print(g_h0.get_shape().as_list())
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
#    print(g_h4.get_shape().as_list())
    
    return g_h4


def discriminator(x):

    x = tf.reshape(x, [batch_size, img_width, img_height, color])

    d_h0 = conv2d(input=x,
                  output_depth=d_depth[0],
                  name='disc_vars')

#    print(d_h0.get_shape().as_list())

    d_h1 = conv2d(input=d_h0,
                  output_depth=d_depth[1],
                  name='disc_vars')
    d_h1 = batch_normalization_and_relu(d_h1, "disc")

#    print(d_h1.get_shape().as_list())

    d_h2 = conv2d(input=d_h1,
                  output_depth=d_depth[2],
                  name='disc_vars')
    d_h2 = batch_normalization_and_relu(d_h2, "disc")

#    print(d_h2.get_shape().as_list())

    d_h3 = conv2d(input=d_h2,
                  output_depth=d_depth[3],
                  name='disc_vars')

#    print(d_h3.get_shape().as_list())

    d_h3_shape = d_h3.get_shape().as_list()
    d_h3 = tf.reshape(d_h3, [batch_size, d_h3_shape[1]*d_h3_shape[2]*d_h3_shape[3]])

    print(d_h3.get_shape().as_list())

    output = mat_operation(d_h3, 1, name='disc_vars')
    output = tf.nn.sigmoid(output)

    return output


def train():
    # placeholder
    
    gen_input = tf.placeholder(tf.float32, [None, noise_dim])
    disc_input = tf.placeholder(tf.float32, [None, img_width, img_height, color])

    gen_output = generator(gen_input)

#    print(gen_output.get_shape().as_list())
    fake_output = discriminator(gen_output)
    real_output = discriminator(disc_input)


    disc_loss = -tf.reduce_mean(tf.log(real_output) + tf.log(1. - fake_output))
    gen_loss = -tf.reduce_mean(tf.log(fake_output))


    tvar = tf.trainable_variables()
    gvar = [var for var in tvar if 'gen' in var.name]
    dvar = [var for var in tvar if 'disc' in var.name]

    gen_train_step = tf.train.AdamOptimizer(learning_rate).minimize(gen_loss, var_list=gvar)
    disc_train_step = tf.train.AdamOptimizer(learning_rate).minimize(disc_loss, var_list=dvar)
    num_img = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("train start!")
        for i in range(epoch):
           # for j in range(1):
            for j in range(int(data_length/batch_size)):
                batch_data = data[j*batch_size:(j+1)*batch_size]
                d_loss, _ = sess.run([disc_loss, disc_train_step], feed_dict={disc_input:batch_data, gen_input:np.random.uniform(-1., 1., [batch_size, noise_dim])})
                g_loss, _ = sess.run([gen_loss, gen_train_step], feed_dict={gen_input: np.random.uniform(-1., 1., [batch_size, noise_dim])})

            print("epoch : ", i, "discriminator_loss :", d_loss, "generator_loss", g_loss)

            images = sess.run(gen_output, feed_dict={gen_input : np.random.uniform(-1., 1., [batch_size, noise_dim])})
#            images = tf.summary.image("G", images)
            path = 'generated_image/%s.png' % str(num_img)
            img_save(images, path)
            num_img += 1

"""            
            print(images)
            fig = plot(images)
            plt.savefig('generated_image/%s.png' % str(num_img).zfill(3), bbox_inches='tight')
            num_img += 1
            plt.close(fig)
"""

if __name__ == "__main__":
    train()
