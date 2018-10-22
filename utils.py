import numpy as np
#import cPickle
import os

from scipy.misc import imsave

def img_save(images, path):
    image = images[0]
    imsave(path, image)

def load_mnist():
    data_dir = 'mnist'
    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    data = loaded[16:].reshape((60000,28,28,1)).astype(np.float)
    data = add_zero_padding(data)

    return data

def load_cifar():
    dir_path = 'cifar-10'
    data = []

    for i in range(1,6):
        file_name = "data_batch_" + str(i)
        file_path = os.path.join(dir_path, file_name)
        with open(file_path, 'rb') as file:
            image_dict = cPickle.load(file)

        image = image_dict["data"]
        for img in image:
            img = img.reshape(32,32,3)
            data.append(img)

    return(np.array(data))

def add_zero_padding(data):
    padding = np.zeros((60000, 32, 32, 1))
    padding[:,:28,:28,:] = data
    return padding
