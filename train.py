iport tensorflow as tf
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

from ops import *
from utils import *

if os.environ.get('DISPLAY', '')== '':
        plt.switch_backend('agg')


print("image loading start")

class Model(object):

    def __init__(self, data_name):

        self.data_name = data_name
        self.ckpt_path = 'checkpoint'

        if(data_name == 'cifar-10'):
            self.data = load_cifar()
            self.rgb = True
            self.ckpt_name = 'cifar_checkpoint.ckpt'

        elif(data_name == 'mnist'):
            self.data = load_mnist()
            self.rgb = False
            self.ckpt_name = 'mnist_checkpoint.ckpt'
        
        print("image is loaded")

        self.ckpt_path = os.path.join(self.ckpt_path, self.ckpt_name)
        print(self.ckpt_path)

        self.img_width = 64
        self.img_heigth = 64
        
        if self.rgb is True:
            self.img_depth = 3
        else:
            self.img_depth = 1

        self.g_depth = [1024, 512, 256, 128, self.img_depth]
        self.g_length = [4, 8, 16, 32, 64]
        self.d_depth = [128, 256, 512, 1024, 1]

        self.data_length = len(self.data)

        self.nosie_dim = 100
        self.batch_size = 100
        self.learning_rate = 0.0002
        self.epoch = 20

        #mnist = input_data.read_data_sets("data/", one_hot=True)

    def generator(self, Z,reuse=False, isTrain=True):

        with tf.variable_scope('gen', reuse=reuse):

            # 1st hidden layer
            conv1 = tf.layers.conv2d_transpose(Z, self.g_depth[0], [4, 4], strides=(1, 1), padding='valid')
            conv1 = batch_normalization_and_relu(conv1, "gen")

            # 2nd hidden layer
            conv2 = tf.layers.conv2d_transpose(conv1, self.g_depth[1], [4, 4], strides=(2, 2), padding='same')
            conv2 = batch_normalization_and_relu(conv2, "gen")

            # 3rd hidden layer
            conv3 = tf.layers.conv2d_transpose(conv2, self.g_depth[2], [4, 4], strides=(2, 2), padding='same')
            conv3 = batch_normalization_and_relu(conv3, "gen")

            # 4th hidden layer
            conv4 = tf.layers.conv2d_transpose(conv3, self.g_depth[3], [4, 4], strides=(2, 2), padding='same')
            conv4 = batch_normalization_and_relu(conv4, "gen")

            # output layer
            conv5 = tf.layers.conv2d_transpose(conv4, self.g_depth[4], [4, 4], strides=(2, 2), padding='same')
            output = tf.nn.tanh(conv5)
        return output


    def discriminator(self, x, reuse=False,isTrain=True):

        with tf.variable_scope('disc', reuse=reuse):
        # 1st hidden layer
            conv1 = tf.layers.conv2d(x, self.d_depth[0], [4, 4], strides=(2, 2), padding='same')
            conv1 = lrelu(conv1, 0.2)

            # 2nd hidden layer
            conv2 = tf.layers.conv2d(conv1, self.d_depth[1], [4, 4], strides=(2, 2), padding='same')
            conv2 = batch_normalization_and_relu(conv2, "disc")

            # 3rd hidden layer
            conv3 = tf.layers.conv2d(conv2, self.d_depth[2], [4, 4], strides=(2, 2), padding='same')
            conv3 = batch_normalization_and_relu(conv3, "disc")

            # 4th hidden layer
            conv4 = tf.layers.conv2d(conv3, self.d_depth[3], [4, 4], strides=(2, 2), padding='same')
            conv4 = batch_normalization_and_relu(conv4, "disc")

            # output layer
            conv5 = tf.layers.conv2d(conv4, self.d_depth[4], [4, 4], strides=(1, 1), padding='valid')
            o = tf.nn.sigmoid(conv5)

        return o, conv5

    def train(self):
        # placeholder
        
        gen_input = tf.placeholder(tf.float32, [None, 1, 1, self.nosie_dim])
        disc_input = tf.placeholder(tf.float32, [None, self.img_width, self.img_heigth, self.img_depth])
        batch_ = tf.placeholder(tf.int32, None)

        gen_output = self.generator(gen_input)
   
        real_output, real_output_logits = self.discriminator(disc_input)
        fake_output, fake_output_logits = self.discriminator(gen_output,reuse=True)

        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_output_logits, labels=tf.ones([batch_, 1, 1, 1])))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output_logits, labels=tf.zeros([batch_, 1, 1, 1])))
        disc_loss = D_loss_real + D_loss_fake
        gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output_logits, labels=tf.ones([batch_, 1, 1, 1])))

        tvar = tf.trainable_variables()
        gvar = [var for var in tvar if 'gen' in var.name]
        dvar = [var for var in tvar if 'disc' in var.name]

        gen_train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(gen_loss, var_list=gvar)
        disc_train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(disc_loss, var_list=dvar)
        num_img = 1

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print("train start!")
            for i in range(self.epoch):
               # for j in range(1):
                for j in range(int(self.data_length/self.batch_size)):
                    batch_data = self.data[j*self.batch_size:(j+1)*self.batch_size]
                    Z = np.random.uniform(-1., 1., [self.batch_size,1, 1, self.nosie_dim])
                    d_loss, _ = sess.run([disc_loss, disc_train_step], feed_dict={disc_input:batch_data, gen_input: Z, batch_ : self.batch_size})

                    Z = np.random.uniform(-1., 1., [self.batch_size,1, 1, self.nosie_dim])
                    g_loss, _ = sess.run([gen_loss, gen_train_step], feed_dict={gen_input: Z, batch_ : self.batch_size})

                np.random.shuffle(self.data)
                
                batch_data = self.data[:64, :,:,:]
                Z = np.random.uniform(-1., 1., [64,1, 1, self.nosie_dim])
                d_loss = sess.run([disc_loss], feed_dict={disc_input:batch_data, gen_input: Z, batch_ : 64})
                g_loss = sess.run([gen_loss], feed_dict={gen_input: Z, batch_ : 64})

                print("epoch : {:2d}, discriminaator_loss : {:.2f}, generator_loss : {:.2f}".format(i+1, d_loss[0], g_loss[0]))

                images = sess.run(gen_output, feed_dict={gen_input : Z})
                path = 'generated_image/%s.png' % str(num_img).zfill(3)
#                img_save(images, path)
                fig = plot(images)
                plt.savefig(path, bbox_inches='tight')
                num_img += 1
                plt.close(fig)
            saver.save(sess, self.ckpt_path)


if __name__ == "__main__":
    print(sys.argv[1])
    model = Model(sys.argv[1])
    model.train()
    print(np.shape(model.data))

