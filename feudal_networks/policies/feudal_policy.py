
import distutils.version
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

import feudal_networks.policies.policy as policy
import feudal_networks.policies.policy_utils as policy_utils
from feudal_networks.models.models import SingleStepLSTM


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

    def _build_model(self):
        """
        Builds the manager and worker models.
        """
        with tf.variable_scope('FeUdal'):
            self._build_placeholders()
            self._build_perception()
            self._build_manager()
            self._build_worker()
            self._build_loss()
        self.var_list = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'FeUdal')

        # for v in self.var_list:
        #     print v

    def _build_placeholders(self):
        self.obs = tf.placeholder(tf.float32, [None] + list(self.obs_space))
        self.prev_g = tf.placeholder(tf.float32, (None,self.c-1,self.g_dim))
        self.R = tf.placeholder(tf.float32,(None,))
        self.Ri = tf.placeholder(tf.float32,(None,))
        self.s_diff = tf.placeholder(tf.float32,(None,self.g_dim))
        self.ac = tf.placeholder(tf.float32,(None,self.act_space))

    def _build_perception(self):
        conv1 = tf.layers.conv2d(inputs=self.obs,
                                filters=16,
                                kernel_size=[8, 8],
                                activation=tf.nn.elu,
                                strides=4)
        conv2 = tf.layers.conv2d(inputs=conv1,
                                filters=32,
                                kernel_size=[4,4],
                                activation=tf.nn.elu,
                                strides=2)

        flattened_filters = policy_utils.flatten(conv2)
        self.z = tf.layers.dense(inputs=flattened_filters,\
                                units=256,\
                                activation=tf.nn.relu)

    def _build_manager(self):
        with tf.variable_scope('manager'):
            # Calculate manager internal state
            self.s = tf.layers.dense(inputs=self.z,\
                                            units=self.g_dim,\
                                            activation=tf.nn.relu)

            # Calculate manager output g
            x = tf.expand_dims(self.s, [0])
            self.manager_lstm = SingleStepLSTM(x,\
                                                self.g_dim,\
                                                step_size=tf.shape(self.obs)[:1])
            g_hat = self.manager_lstm.output
            self.g = tf.nn.l2_normalize(g_hat, dim=1)

            self.manager_vf = self._build_value(g_hat)

    def _build_worker(self):
        with tf.variable_scope('worker'):
            num_acts = self.act_space

            # Calculate U
            self.worker_lstm = SingleStepLSTM(tf.expand_dims(self.z, [0]),\
                                                size=num_acts * self.k,
                                                step_size=tf.shape(self.obs)[:1])
            flat_logits = self.worker_lstm.output

            self.worker_vf = self._build_value(flat_logits)

            U = tf.reshape(flat_logits,[-1,num_acts,self.k])

            # Calculate w
            cut_g = tf.expand_dims(tf.stop_gradient(self.g), [1])
            gstack = tf.concat([self.prev_g,cut_g], axis=1)
            gsum = tf.reduce_sum(gstack, axis=1)
            phi = tf.get_variable("phi", (self.g_dim, self.k))
            w = tf.matmul(gsum,phi)
            w = tf.expand_dims(w,[2])
            # Calculate policy and sample
            logits = tf.reshape(tf.matmul(U,w),[-1,num_acts])
            self.pi = tf.nn.softmax(logits)
            self.sample = policy_utils.categorical_sample(
                tf.reshape(logits,[-1,num_acts]), num_acts)[0, :]

    def _build_value(self,input):
        with tf.variable_scope('VF'):
            hidden = tf.layers.dense(inputs=input,\
                                units=self.config.vf_hidden_size,\
                                activation=tf.nn.relu)

            w = tf.get_variable("weights", (self.config.vf_hidden_size, 1))
            return tf.matmul(hidden,w)

    def _build_loss(self):
        cutoff_vf_manager = tf.reshape(tf.stop_gradient(self.manager_vf),[-1])
        dot = tf.reduce_sum(tf.multiply(self.s_diff,self.g ),axis=1)
        mag = tf.norm(self.s_diff,axis=1)*tf.norm(self.g,axis=1)
        dcos = dot/mag
        manager_loss = -tf.reduce_sum((self.R-cutoff_vf_manager)*dcos)

        cutoff_vf_worker = tf.reshape(tf.stop_gradient(self.worker_vf),[-1])
        log_p = tf.reduce_sum(tf.log(self.pi)*self.ac,1)
        worker_loss = (self.R + self.config.alpha*self.Ri - cutoff_vf_worker)*log_p
        worker_loss = -tf.reduce_sum(worker_loss,axis=0)

        Am = self.R-self.manager_vf
        manager_vf_loss = .5*tf.reduce_sum(tf.square(Am))

        Aw = (self.R + self.config.alpha*self.Ri)-self.worker_vf
        worker_vf_loss = .5*tf.reduce_sum(tf.square(Aw))

        entropy = - tf.reduce_sum(self.pi * log_p)

        self.loss = worker_loss+manager_loss+\
                    worker_vf_loss + manager_vf_loss-\
                    entropy*.01

        print self.loss

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
