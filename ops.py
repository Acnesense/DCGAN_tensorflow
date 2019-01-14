import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



def batch_normalization_and_relu(layer, name):
    if(name == "gen"):
        output = tf.nn.relu(tf.layers.batch_normalization(layer, training=True))

    elif(name == "disc"):
        output = lrelu(tf.layers.batch_normalization(layer, training=True))

    return output


def mat_operation(input, output_size, name):

    input_shape = input.get_shape().as_list()
    shape = [input_shape[-1], output_size]
    with tf.variable_scope(name):

        W = tf.Variable(tf.random_normal(shape=shape, stddev=5e-2))
        b = tf.Variable(tf.constant(0.1, shape=[output_size]))

    output = tf.matmul(input, W) + b

    return output


def sigmoid_cross_entropy_with_logits(x, y):
    try:
        return tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=y)
    except:
        return tf.nn.sigmoid_cross_entropy_with_logits(labels=x, target=y)


def lrelu(input, leak=0.2):
    return tf.maximum(input, leak*input)


def plot(samples):
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        plt.subplot(gs[i])
        plt.axis('off')
        plt.imshow(sample.reshape(64, 64), cmap='gray')
    return fig
