import tensorflow as tf
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
#import cPickle
from ops import *
from utils import *

if os.environ.get('DISPLAY', '')== '':
        plt.switch_backend('agg')


print("image loading start")

class Model(object):

    def __init__(self, data_name):

        self.data_name = data_name

        if(data_name == 'cifar-10'):
            self.data = load_cifar()
            self.rgb = True

        elif(data_name == 'mnist'):
            self.data = load_mnist()
            self.rgb = False           
            print("image is loaded")

        self.img_width = 64
        self.img_heigth = 64

        if self.rgb is True:
            self.color = 3
        else:
            self.color = 1

        self.g_depth = [1024, 512, 256, 128, self.color]
        self.g_length = [4, 8, 16, 32, 64]
        self.d_depth = [64, 128, 256, 512, 1024]



        self.data_length = len(self.data)

        self.nosie_dim = 100
        self.batch_size = 200
        self.learning_rate = 0.0002
        self.epoch = 100

        #mnist = input_data.read_data_sets("data/", one_hot=True)

    def generator(self, Z):

#        print(Z.get_shape().as_list())
        g_h0 = mat_operation(Z, self.g_length[0]*self.g_length[0]*self.g_depth[0], 'gen_vars')
        g_h0 = tf.reshape(g_h0, [-1,self.g_length[0],self.g_length[0], self.g_depth[0]])

#        print(g_h0.get_shape().as_list())
        g_h1 = conv2d_transpose(input=g_h0,
                                output_shape=[self.batch_size, self.g_length[1], self.g_length[1], self.g_depth[1]], 
                                name='gen_var')
#        print(g_h1.get_shape().as_list())

        g_h2 = conv2d_transpose(input=g_h1,
                                output_shape=[self.batch_size, self.g_length[2], self.g_length[2], self.g_depth[2]], 
                                name='gen_var')
        g_h2 = batch_normalization_and_relu(g_h2, "gen")
        
#        print(g_h2.get_shape().as_list())

        g_h3 = conv2d_transpose(input=g_h2,
                                output_shape=[self.batch_size, self.g_length[3], self.g_length[3], self.g_depth[3]], 
                                name='gen_var')
        g_h3 = batch_normalization_and_relu(g_h3, "gen")
        
#        print(g_h3.get_shape().as_list())

        g_h4 = conv2d_transpose(input=g_h3,
                                output_shape=[self.batch_size, self.g_length[4], self.g_length[4], self.g_depth[4]], 
                                name='gen_var')
        g_h4 = tf.nn.tanh(g_h4)
#        print(g_h4.get_shape().as_list())
        
        return g_h4


    def discriminator(self, x):

        x = tf.reshape(x, [self.batch_size, self.img_width, self.img_heigth, self.color])

        d_h0 = conv2d(input=x,
                    output_depth=self.d_depth[0],
                    name='disc_vars')

    #    print(d_h0.get_shape().as_list())

        d_h1 = conv2d(input=d_h0,
                    output_depth=self.d_depth[1],
                    name='disc_vars')
        d_h1 = batch_normalization_and_relu(d_h1, "disc")

    #    print(d_h1.get_shape().as_list())

        d_h2 = conv2d(input=d_h1,
                    output_depth=self.d_depth[2],
                    name='disc_vars')
        d_h2 = batch_normalization_and_relu(d_h2, "disc")

    #    print(d_h2.get_shape().as_list())

        d_h3 = conv2d(input=d_h2,
                    output_depth=self.d_depth[3],
                    name='disc_vars')

    #    print(d_h3.get_shape().as_list())

        d_h3_shape = d_h3.get_shape().as_list()
        d_h3 = tf.reshape(d_h3, [self.batch_size, d_h3_shape[1]*d_h3_shape[2]*d_h3_shape[3]])

    #    print(d_h3.get_shape().as_list())

        output = mat_operation(d_h3, 1, name='disc_vars')
        output = tf.nn.sigmoid(output)

        return output

    def train(self):
        # placeholder
        
        gen_input = tf.placeholder(tf.float32, [None, self.nosie_dim])
        disc_input = tf.placeholder(tf.float32, [None, self.img_width, self.img_heigth, self.color])

        gen_output = self.generator(gen_input)

    #    print(gen_output.get_shape().as_list())
        fake_output = self.discriminator(gen_output)
        real_output = self.discriminator(disc_input)

        disc_loss = -tf.reduce_mean(tf.log(real_output) + tf.log(1. - fake_output))
        gen_loss = -tf.reduce_mean(tf.log(fake_output))

        tvar = tf.trainable_variables()
        gvar = [var for var in tvar if 'gen' in var.name]
        dvar = [var for var in tvar if 'disc' in var.name]

        gen_train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(gen_loss, var_list=gvar)
        disc_train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(disc_loss, var_list=dvar)
        num_img = 0

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print("train start!")
            for i in range(self.epoch):
               # for j in range(1):
                for j in range(int(self.data_length/self.batch_size)):
                    batch_data = self.data[j*self.batch_size:(j+1)*self.batch_size]
                    d_loss, _ = sess.run([disc_loss, disc_train_step], feed_dict={disc_input:batch_data, gen_input:np.random.uniform(-1., 1., [self.batch_size, self.nosie_dim])})
                    g_loss, _ = sess.run([gen_loss, gen_train_step], feed_dict={gen_input: np.random.uniform(-1., 1., [self.batch_size, self.nosie_dim])})

                print("self.epoch : ", i, "discriminator_loss :", d_loss, "generator_loss", g_loss)
                images = sess.run(gen_output, feed_dict={gen_input : np.random.uniform(-1., 1., [self.batch_size, self.nosie_dim])})
                images = tf.summary.image("G", images)
                path = 'generated_image/%s.png' % str(num_img)
#                img_save(images, path)

    """            
                print(images)
                fig = plot(images)
                plt.savefig('generated_image/%s.png' % str(num_img).zfill(3), bbox_inches='tight')
                num_img += 1
                plt.close(fig)
    """

if __name__ == "__main__":
    print(sys.argv[1])
    model = Model(sys.argv[1])
    print(np.shape(model.data))
