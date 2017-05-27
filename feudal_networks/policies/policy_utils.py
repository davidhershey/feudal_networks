
import numpy as np
import tensorflow as tf

def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

def categorical_sample(logits, d):
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(
        logits, [1], keep_dims=True), 1), [1])
    return tf.one_hot(value, d)