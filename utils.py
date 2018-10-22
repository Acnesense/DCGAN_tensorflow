import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from scipy.misc import imsave

def img_save(images, path):
    image = images[0]
    imsave(path, image)

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

