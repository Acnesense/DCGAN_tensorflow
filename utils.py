import numpy as np
#import cPickle
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.misc import imsave

try:
    import cPickle
except:
    import pickle

sess = tf.Session()


def img_save(images, path):
    print(np.shape(images))

    image = images[0]
    imsave(path, image)


def load_mnist():
    
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=[])
    train_set = tf.image.resize_images(mnist.train.images, [64,64]).eval(session=tf.Session())
    train_set = (train_set - 0.5) / 0.5

    return train_set


def load_cifar():
    dir_path = 'cifar-10'
    data = []

    for i in range(1,6):
        file_name = "data_batch_" + str(i)
        file_path = os.path.join(dir_path, file_name)
        with open(file_path, 'rb') as file:
            try:
                image_dict = cPickle.load(file)
                images = image_dict["data"]

            except:
                image_dict = pickle.load(file, encoding='bytes')
                images = image_dict[b'data']

        
        for img in images:
            img = np.reshape(img, [32,32,3])
            img = tf.image.resize_images(img, [64,64]).eval(session=tf.Session())
            img = (img / 127.5) -1 
            data.append(img)

    return np.array(data)


def add_zero_padding(data):
    padding = np.zeros((60000, 32, 32, 1))
    padding[:,:28,:28,:] = data
    return padding

def plot(samples):
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        plt.imshow(sample.reshape(64, 64))
    return fig

if __name__ == "__main__":
    load_mnist()
