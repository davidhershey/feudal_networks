import policy
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import numpy as np
import distutils.version
use_tf100_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('1.0.0')

def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

class FeudalBatchProcessor(object):
    """
    This class adapts the batch of PolicyOptimizer to a batch useable by
    the FeudalPolicy.
    """
    def __init__(self):
        pass

    def process_batch(self, batch):
        """
        Converts a normal batch into one used by the FeudalPolicy update.

        FeudalPolicy requires a batch of the form:

        c previous timesteps - batch size timesteps - c future timesteps

        This class handles the tracking the leading and following timesteps over
        time. Additionally, it also computes values across timesteps from the
        batch to provide to FeudalPolicy.
        """
        pass

class FeudalPolicy(policy.Policy):
    """
    Policy of the Feudal network architecture.
    """

    def __init__(self, obs_space, act_space, config):
        self.obs_space = obs_space
        self.act_space = act_space
        self.config = config
        self.k = 16 #Dimensionality of w
        self.g_dim = 256
        self.c = 10
        self.state_in = {}
        self.state_out = {}
        self.state_init= {}
        self._build_model()

    def _build_placeholders(self):
        self.obs = tf.placeholder(tf.float32, [None] + list(self.obs_space))
        self.prev_g = tf.placeholder(tf.float32, (None,self.c-1,self.g_dim))

    def _build_model(self):
        """
        Builds the manager and worker models.
        """
        self._build_placeholders()
        self._build_perception()
        self._build_manager()
        self._build_worker()

    def _build_perception(self):
        conv1 = tf.layers.conv2d(inputs=self.obs,
                                filters=16,
                                kernel_size=[8, 8],
                                activation=tf.nn.relu,
                                strides=4)
        conv2 = tf.layers.conv2d(inputs=conv1,
                                filters=32,
                                kernel_size=[4,4],
                                activation=tf.nn.relu,
                                strides=2)

        flattened_filters = flatten(conv2)
        self.z = tf.layers.dense(inputs=flattened_filters,\
                                units=256,\
                                activation=tf.nn.relu)

    def _build_manager(self):
        with tf.variable_scope('manager'):
            self.manager_s = tf.layers.dense(inputs=self.z,\
                                            units=256,\
                                            activation=tf.nn.relu)

            x = tf.expand_dims(self.manager_s, [0])


            g_hat =self._build_lstm(x,self.g_dim,'manager')
            self.g = tf.nn.l2_normalize(g_hat,dim=1)


    def _build_worker(self):
        with tf.variable_scope('worker'):
            num_acts = np.prod(list(self.act_space))
            flat_logits =self._build_lstm(tf.expand_dims(self.z, [0]),\
                                    size=num_acts*self.k,\
                                    name='worker')
            print flat_logits
            U = tf.reshape(flat_logits,[-1,num_acts,self.k])

            cut_g = tf.expand_dims(tf.stop_gradient(self.g),[1])
            gstack = tf.concat([self.prev_g,cut_g],axis=1)
            gsum = tf.reduce_sum(gstack,axis=1)
            phi = tf.get_variable("phi", (self.g_dim,self.k))
            w = tf.matmul(gsum,phi)
            w = tf.expand_dims(w,[2])
            logits = tf.matmul(U,w)
            self.pi = tf.nn.softmax(logits)
            print self.pi.get_shape()



    def _build_lstm(self,x,size,name):
        if use_tf100_api:
            lstm = rnn.BasicLSTMCell(size, state_is_tuple=True)
        else:
            lstm = rnn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True)

        step_size = tf.shape(self.obs)[:1]

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init[name] = [c_init, h_init]

        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
        self.state_in[name] = [c_in, h_in]

        if use_tf100_api:
            state_in = rnn.LSTMStateTuple(c_in, h_in)
        else:
            state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)

        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=step_size,
            time_major=False)

        lstm_c, lstm_h = lstm_state
        self.state_out[name] = [lstm_c[:1, :], lstm_h[:1, :]]
        # print name, lstm_outputs.get_shape()
        return tf.reshape(lstm_outputs, [-1, size])

    def _build_loss(self):
        pass

    def act(self, obs, prev_internal):
        pass

    def value(self, obs, prev_internal):
        pass

    def update(self, sess, train_op, batch):
        """
        This function performs a weight update. It does this by first converting
        the batch into a feudal_batch using a FeudalBatchProcessor. The
        feudal_batch can then be passed into a session run call directly.
        """
        pass
