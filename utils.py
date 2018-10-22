import numpy as np
import cPickle
import os

from scipy.misc import imsave

def img_save(images, path):
    image = images[0]
    imsave(path, image)

def load_mnist():
    
    return 0

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

def mnist():
    
    return 0
"""
def img_save(images):
    fig = plt.figure()
    fig = plt.figure(figsize=(8,8))
    gs = gridspec.GridSpec(8,8)
    gs.update(wspace=0.05, hspace=0.05)

    for i, image in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        
        plt.imshow(image.reshape(32,32,3))
    return fig
"""

