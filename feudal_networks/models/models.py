
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size],
        initializer=initializer)
    b = tf.get_variable(name + "/b", [size],
        initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b

def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME",
        dtype=tf.float32, collections=None):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1],
            int(x.get_shape()[3]), num_filters]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[:3])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = np.prod(filter_shape[:2]) * num_filters
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype,
            tf.random_uniform_initializer(-w_bound, w_bound),
            collections=collections)
        # adding initialization to bias because otherwise the network will
        # output all zeros, which is normally fine, but in the feudal case
        # this yields a divide by zero error. Bounds are just small random.
        b = tf.get_variable("b", [1, 1, 1, num_filters],
            initializer=tf.random_uniform_initializer(-w_bound, w_bound),
            collections=collections)
        return tf.nn.conv2d(x, w, stride_shape, pad) + b

def build_lstm(x, size, name, step_size):
    lstm = rnn.BasicLSTMCell(size, state_is_tuple=True)

    c_init = np.zeros((1, lstm.state_size.c), np.float32)
    h_init = np.zeros((1, lstm.state_size.h), np.float32)
    state_init = [c_init, h_init]

    c_in = tf.placeholder(tf.float32,
            shape=[1, lstm.state_size.c],
            name='c_in')
    h_in = tf.placeholder(tf.float32,
            shape=[1, lstm.state_size.h],
            name='h_in')
    state_in = [c_in, h_in]

    state_in = rnn.LSTMStateTuple(c_in, h_in)

    lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
        lstm, x, initial_state=state_in, sequence_length=step_size,
        time_major=False)
    lstm_outputs = tf.reshape(lstm_outputs, [-1, size])

    lstm_c, lstm_h = lstm_state
    state_out = [lstm_c[:1, :], lstm_h[:1, :]]
    return lstm_outputs, state_init, state_in, state_out

class SingleStepLSTM(object):

    def __init__(self, x, size, step_size):
        lstm = rnn.BasicLSTMCell(size, state_is_tuple=True)

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]

        c_in = tf.placeholder(tf.float32,
                shape=[1, lstm.state_size.c],
                name='c_in')
        h_in = tf.placeholder(tf.float32,
                shape=[1, lstm.state_size.h],
                name='h_in')
        self.state_in = [c_in, h_in]

        state_in = rnn.LSTMStateTuple(c_in, h_in)

        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=step_size,
            time_major=False)
        lstm_outputs = tf.reshape(lstm_outputs, [-1, size])

        lstm_c, lstm_h = lstm_state
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
        self.output = lstm_outputs

def conditional_backprop(do_backprop, tensor):
    # do_backprop = tf.Print(do_backprop, [do_backprop], "switch query")
    t = tf.cond(tf.cast(do_backprop, tf.bool),
                lambda: tensor,
                lambda: tf.zeros_like(tensor))
    y = t + tf.stop_gradient(tensor - t)
    return y

def DilatedLSTM(s_t, size,state_in,idx_in,chunks=8):

    lstm = rnn.LSTMCell(size, state_is_tuple=True)
    c_init = np.zeros((chunks, lstm.state_size.c), np.float32)
    h_init = np.zeros((chunks, lstm.state_size.h), np.float32)
    state_init = [c_init, h_init]

    def dlstm_scan_fn(previous_output, current_input):
        i = previous_output[2]
        c = previous_output[1][0]
        h = previous_output[1][1]
        # old_state = [tf.expand_dims(c[i],[0]),tf.expand_dims(h[i],[0])]
        old_state = rnn.LSTMStateTuple(tf.expand_dims(c[i],[0]),tf.expand_dims(h[i],[0]))
        out, state_out = lstm(current_input, old_state)

        co = state_out[0] - c[i]
        ho = state_out[1] - h[i]

        cob = tf.concat([tf.zeros((i,size)),co,tf.zeros((chunks-i-1,size))],axis=0)
        hob = tf.concat([tf.zeros((i,size)),ho,tf.zeros((chunks-i-1,size))],axis=0)

        # cob = tf.Print(cob,[tf.shape(cob)])
        c_out = c + cob
        h_out = h + hob
        state_out = [c_out,h_out]
        i += tf.constant(1)
        new_i = tf.mod(i, chunks)
        out = tf.reduce_mean(h_out,axis=0)
        return out, state_out, new_i

    rnn_outputs, final_states, out_idx = tf.scan(dlstm_scan_fn,
                                                  tf.transpose(s_t, [1, 0, 2]),
                                                  initializer=(state_in[1][0], state_in, idx_in))


    state_out = [final_states[0][0, :, :], final_states[1][0, :, :]]
    # cell_states = final_states[0][:, 0, :]
    # out_states = final_states[1][:, 0, :]
    return rnn_outputs, state_init, state_out,out_idx
