import tensorflow as tf

def conv2d_transpose(input, output_shape, name ,k_h=5, k_w=5):
    
    input_shape = input.get_shape().as_list()
    filter_shape = [k_h, k_w, output_shape[-1], input_shape[-1]]

    with tf.variable_scope(name):

        W = tf.Variable(tf.random_normal(shape=filter_shape, stddev=5e-2))
        output = tf.nn.conv2d_transpose(input, W, output_shape = output_shape, strides=[1,2,2,1])

    return output


def conv2d(input, output_depth,name, k_h=5, k_w=5):
    input_shape = input.get_shape().as_list()
    filter_shape = [k_h, k_w, input_shape[-1], output_depth]

    with tf.variable_scope(name):

        W = tf.Variable(tf.random_normal(shape=filter_shape, stddev=5e-2))
        b = tf.Variable(tf.constant(0.1, shape=[output_depth]))
        output = tf.nn.conv2d(input,W , strides=[1,2,2,1], padding="SAME")+b
        
    return output
    

def batch_normalization_and_relu(layer, name):
    if(name == "gen"):
        output = tf.nn.relu(tf.layers.batch_normalization(layer, training=True))

    elif(name == "disc"):
        output = leaky_relu(tf.layers.batch_normalization(layer, training=True))

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


def leaky_relu(input, leak=0.2 ):
    return tf.maximum(input, leak*input)



