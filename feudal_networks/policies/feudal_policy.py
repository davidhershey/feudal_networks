
import distutils.version
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

import feudal_networks.policies.policy as policy
import feudal_networks.policies.policy_utils as policy_utils
from feudal_networks.models.models import (
    SingleStepLSTM, DilatedLSTM, conv2d, linear, normalized_columns_initializer
)
from feudal_networks.policies.feudal_batch_processor import FeudalBatchProcessor


class FeudalPolicy(policy.Policy):
    """
    Policy of the Feudal network architecture.
    """

    def __init__(self, obs_space, act_space, global_step, config):
        self.obs_space = obs_space
        self.act_space = act_space
        self.global_step = global_step
        self.config = config
        self.k = config.k # dimensionality of w
        self.g_dim = config.g_dim # dimensionality of goals
        self.batch_processor = FeudalBatchProcessor(config.c)
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
                tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def _build_placeholders(self):
        # standard for all policies
        self.obs = tf.placeholder(tf.float32, 
            shape=[None] + list(self.obs_space),
            name='obs')
        self.r = tf.placeholder(tf.float32,
            shape=(None,),
            name='returns')
        self.ac = tf.placeholder(tf.float32,
            shape=(None,self.act_space),
            name='action_mask')

        # specific to FeUdal
        self.prev_g = tf.placeholder(tf.float32, 
            shape=(None, None, self.g_dim),
            name='goals')
        self.ri = tf.placeholder(tf.float32,
            shape=(None,),
            name='intrinsic_rewards')
        self.s_diff = tf.placeholder(tf.float32,
            shape=(None, self.g_dim),
            name='state_diffs')

    def _build_perception(self):
        x = self.obs
        for i in range(self.config.n_percept_hidden_layer):
            x = tf.nn.elu(conv2d(x, self.config.n_percept_filters,
                "l_{}".format(i + 1), [3, 3], [2, 2]))
        flattened_filters = policy_utils.flatten(x)
        self.z = tf.layers.dense(
            inputs=flattened_filters,
            units=self.config.z_dim,
            activation=tf.nn.elu,
            name='feudal_z'
        )

    def _build_manager(self):
        with tf.variable_scope('manager'):
            # Calculate manager internal state
            self.s = tf.layers.dense(
                inputs=self.z,
                units=self.g_dim,
                activation=tf.nn.elu,
                name='manager_s'
            )

            # Calculate manager output g
            x = tf.expand_dims(self.s, [0])

            if self.config.verbose:
                x = tf.Print(x, [x], message='\nmanager s: ', summarize=5)

            if self.config.manager_rnn_type == 'dilated':
                self.manager_state_in = [
                    tf.placeholder(
                        shape=(1, self.g_dim), 
                        dtype='float32',
                        name='manger_lstm_in1'),
                    tf.placeholder(
                        shape=(1, self.g_dim), 
                        dtype='float32',
                        name='manger_lstm_in2')
                ]
                g_hat, self.manager_state_init, self.manager_state_out = DilatedLSTM(
                    x, self.g_dim, self.manager_state_in, chunks=self.config.c)

            elif self.config.manager_rnn_type == 'lstm':
                self.manager_lstm = SingleStepLSTM(
                    x, self.g_dim, step_size=tf.shape(self.obs)[:1])
                self.manager_state_in = self.manager_lstm.state_in
                self.manager_state_out = self.manager_lstm.state_out
                self.manager_state_init = self.manager_lstm.state_init
                g_hat = self.manager_lstm.output
            else:
                raise ValueError('invalid config.rnn_type: {}'.format(
                    self.config.manager_rnn_type))

            self.g = tf.nn.l2_normalize(g_hat, dim=1)

            if self.config.verbose:
                self.g = tf.Print(self.g, [self.g, g_hat], 
                    message='\nmanager g and g_hat: ', summarize=5)

            self.manager_vf = self._build_value(g_hat)

            if self.config.verbose:
                self.manager_vf = tf.Print(self.manager_vf, [self.manager_vf],
                    message='\nmanager vf: ', summarize=5)

            # add manager c, h to state in and out
            self.state_in = [
                self.manager_state_in[0],
                self.manager_state_in[1]
            ]
            self.state_out = [
                self.manager_state_out[0],
                self.manager_state_out[1]
            ]

    def _build_worker(self):
        with tf.variable_scope('worker'):
            num_acts = self.act_space

            # Calculate U
            self.worker_lstm = SingleStepLSTM(
                tf.expand_dims(self.z, [0]),
                size=self.config.worker_lstm_size,
                step_size=tf.shape(self.obs)[:1]
            )
            flat_logits = self.worker_lstm.output
            flat_logits = tf.layers.dense(
                inputs=flat_logits,
                units=num_acts * self.k,
                activation=None,
                name='flat_logits'
            )

            self.worker_vf = self._build_value(flat_logits)
            U = tf.reshape(flat_logits, 
                shape=[-1, num_acts, self.k],
                name='U')

            # Calculate w
            cut_g = tf.stop_gradient(self.g)
            cut_g = tf.expand_dims(cut_g, [1])
            gstack = tf.concat([self.prev_g,cut_g], axis=1)

            self.last_c_g = gstack[:,1:]
            gsum = tf.reduce_sum(gstack, axis=1)
            phi = tf.get_variable("phi", (self.g_dim, self.k))
            w = tf.matmul(gsum, phi)
            w = tf.expand_dims(w, [2])

            # calculate policy and sample
            logits = tf.reshape(tf.matmul(U, w),[-1, num_acts])
            self.pi = tf.nn.softmax(logits)
            self.log_pi = tf.nn.log_softmax(logits)
            self.sample = policy_utils.categorical_sample(
                tf.reshape(logits,[-1,num_acts]), num_acts)[0, :]

            # add worker c, h to state in and out
            self.state_in.extend([
                self.worker_lstm.state_in[0],
                self.worker_lstm.state_in[1],
            ])
            self.state_out.extend([
                self.worker_lstm.state_out[0],
                self.worker_lstm.state_out[1],
            ])

    def _build_value(self,input):
        with tf.variable_scope('VF'):
            hidden = tf.layers.dense(
                        inputs=input,
                        units=self.config.vf_hidden_size,
                        activation=tf.nn.elu,
                        name='hidden'
            )
            vf = linear(hidden, 1, 'value', 
                initializer=normalized_columns_initializer(1.0)
            )
            return vf

    def _build_loss(self):
        # manager policy loss
        cutoff_vf_manager = tf.reshape(tf.stop_gradient(self.manager_vf),[-1])
        dot = tf.reduce_sum(tf.multiply(self.s_diff, self.g), axis=1)
        gcut = self.g
        mag = tf.norm(self.s_diff, axis=1) * tf.norm(gcut, axis=1) + self.config.eps
        dcos = dot / mag
        manager_loss = -tf.reduce_sum((self.r - cutoff_vf_manager) * dcos)

        # manager value loss
        Am = self.r-self.manager_vf
        manager_vf_loss = .5*tf.reduce_sum(tf.square(Am))

        # worker policy loss
        cutoff_vf_worker = tf.reshape(tf.stop_gradient(self.worker_vf), [-1])
        log_p = tf.reduce_sum(self.log_pi * self.ac)
        worker_loss = (self.r + self.config.alpha * self.ri - cutoff_vf_worker) * log_p
        worker_loss = -tf.reduce_sum(worker_loss)

        # worker value loss
        Aw = (self.r + self.config.alpha * self.ri) - self.worker_vf
        worker_vf_loss = .5 * tf.reduce_sum(tf.square(Aw))

        entropy = -tf.reduce_sum(self.pi * self.log_pi)

        beta = tf.train.polynomial_decay(
            self.config.beta_start, 
            self.global_step,
            end_learning_rate=self.config.beta_end,
            decay_steps=self.config.decay_steps,
            power=1
        )

        self.loss = (worker_loss + manager_loss + worker_vf_loss + 
                    manager_vf_loss - entropy * beta)

        bs = tf.to_float(tf.shape(self.obs)[0])
        tf.summary.scalar("model/manager_loss", manager_loss / bs)
        tf.summary.scalar("model/worker_loss", worker_loss / bs)
        tf.summary.scalar("model/value_mean", tf.reduce_mean(self.manager_vf))
        tf.summary.scalar("model/value_loss", manager_vf_loss / bs)
        tf.summary.scalar("model/value_loss_scaled", manager_vf_loss / bs * .5)
        tf.summary.scalar("model/entropy", entropy / bs)
        tf.summary.scalar("model/entropy_loss_scaleed", -entropy / bs * beta)
        tf.summary.scalar("model/var_global_norm", 
            tf.global_norm(tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, 
                tf.get_variable_scope().name)))
        tf.summary.scalar("model/beta", beta)
        tf.summary.image("model/state", self.obs)
        self.summary_op = tf.summary.merge_all()

    def get_initial_features(self):
        return np.zeros((1, 1, self.g_dim), np.float32), self.worker_lstm.state_init + self.manager_state_init

    def act(self, ob, g, cw, hw, cm, hm):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.manager_vf, self.g, self.s, self.last_c_g] + self.state_out,
                        {self.obs: [ob], self.state_in[0]: cw, self.state_in[1]: hw,\
                         self.state_in[2]: cm, self.state_in[3]: hm,\
                         self.prev_g: g})

    def value(self, ob, g, cw, hw, cm, hm):
        sess = tf.get_default_session()
        return sess.run(self.manager_vf,
                        {self.obs: [ob], self.state_in[0]: cw, self.state_in[1]: hw,\
                         self.state_in[2]: cm, self.state_in[3]: hm,\
                         self.prev_g: g})[0]

    def update_batch(self, batch):
        return self.batch_processor.process_batch(batch)
