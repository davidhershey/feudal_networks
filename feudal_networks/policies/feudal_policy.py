
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
        self.batch_processor = FeudalBatchProcessor(
            config.c, pad_method=config.batch_pad_method,
            similarity_metric=config.similarity_metric)
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
        self.manager_r = tf.placeholder(tf.float32,
            shape=(None,),
            name='manager_returns')
        self.worker_r = tf.placeholder(tf.float32,
            shape=(None,),
            name='worker_returns')
        self.ac = tf.placeholder(tf.float32,
            shape=(None,self.act_space),
            name='action_mask')

        # specific to FeUdal
        self.prev_g = tf.placeholder(tf.float32,
            shape=(None, self.config.c-1, self.g_dim),
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
            if self.config.use_batch_norm:
                x = tf.contrib.layers.batch_norm(x)
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
            if self.config.s_is_obs:
                self.s = policy_utils.flatten(self.obs)
            else:
                self.s = tf.layers.dense(
                    inputs=self.z,
                    units=self.config.s_dim,
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
                        shape=(1, self.config.manager_lstm_size),
                        dtype='float32',
                        name='manger_lstm_in1'),
                    tf.placeholder(
                        shape=(1, self.config.manager_lstm_size),
                        dtype='float32',
                        name='manger_lstm_in2')
                ]
                self.dilated_idx_in = tf.placeholder(shape=(),dtype='int32')
                lstm_output, self.manager_state_init, self.manager_state_out,self.dilated_idx_out = DilatedLSTM(
                    x,
                    self.config.manager_lstm_size,
                    self.manager_state_in,
                    self.dilated_idx_in,
                    chunks=self.config.c
                )

            elif self.config.manager_rnn_type == 'lstm':
                self.manager_lstm = SingleStepLSTM(
                    x,
                    self.config.manager_lstm_size,
                    step_size=tf.shape(self.obs)[:1]
                )
                self.manager_state_in = self.manager_lstm.state_in
                self.manager_state_out = self.manager_lstm.state_out
                self.manager_state_init = self.manager_lstm.state_init
                lstm_output = self.manager_lstm.output
            else:
                raise ValueError('invalid config.rnn_type: {}'.format(
                    self.config.manager_rnn_type))

            hidden_g_hat = tf.layers.dense(
                        inputs=lstm_output,
                        units=self.config.g_dim,
                        activation=tf.nn.elu,
                        name='hidden_g_hat'
            )
            g_hat = linear(hidden_g_hat, self.config.g_dim, 'g_hat',
                initializer=normalized_columns_initializer(1.0)
            )
            # g_hat = tf.cond(tf.random_uniform(()) < self.config.g_eps,
            #                     lambda: tf.random_normal(tf.shape(g_hat)),
            #                     lambda: g_hat)

            self.g = tf.nn.l2_normalize(g_hat, dim=1)

            if self.config.verbose:
                self.g = tf.Print(self.g, [self.g, g_hat],
                    message='\nmanager g and g_hat: ', summarize=5)

            self.manager_vf = self._build_value(lstm_output)

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

            # Calculate w
            cut_g = tf.stop_gradient(self.g)
            cut_g = tf.expand_dims(cut_g, [1])
            gstack = tf.concat([tf.zeros((tf.shape(cut_g)[0],self.config.c,self.g_dim)),self.prev_g, cut_g], axis=1)

            self.last_c_g = gstack[:,-(self.config.c-1):]
            self.g_sum=gsum = tf.reduce_sum(gstack, axis=1)

            if self.config.verbose:
                gsum = tf.Print(gsum, [gsum],
                    message='\nworker gsum: ', summarize=32)

            phi = tf.get_variable("phi", (self.g_dim, self.k),
                initializer=normalized_columns_initializer(1.))

            if self.config.verbose:
                phi = tf.Print(phi, [phi],
                    message='\nworker phi: ', summarize=5)

            w = tf.matmul(gsum, phi)
            tf.summary.scalar('model/w_magnitude', tf.reduce_mean(w))

            # Calculate U
            self.worker_lstm = SingleStepLSTM(
                tf.expand_dims(self.z, [0]),
                size=self.config.worker_lstm_size,
                step_size=tf.shape(self.obs)[:1]
            )
            lstm_output = self.worker_lstm.output

            if self.config.worker_hint:
                worker_hint = tf.stop_gradient(tf.contrib.layers.batch_norm(w))
                lstm_output = tf.concat([lstm_output, worker_hint], axis=1)

            self.worker_vf = self._build_value(lstm_output)

            flat_logits_hidden = tf.layers.dense(
                inputs=lstm_output,
                units=num_acts * self.k,
                activation=tf.nn.elu,
                name='flat_logits_hidden'
            )

            if self.config.dropout_keep_prob < 1.:
                flat_logits_hidden = tf.nn.dropout(
                    flat_logits_hidden, self.config.dropout_keep_prob)

            flat_logits = tf.layers.dense(
                inputs=flat_logits_hidden,
                units=num_acts * self.k,
                activation=None,
                name='flat_logits'
            )

            self.U = tf.reshape(flat_logits,
                shape=[-1, num_acts, self.k],
                name='U')

            if self.config.verbose:
                self.U = tf.Print(self.U, [self.U],
                    message='\nworker U: ', summarize=num_acts * self.k)

            tf.summary.image('model/U',
                tf.reshape(self.U, (-1, num_acts, self.k, 1)), max_outputs=1)
            tf.summary.image('model/w',
                tf.reshape(w, (-1, 1, self.k, 1)), max_outputs=1)

            # expand dims for combining with U
            w = tf.expand_dims(w, [2])

            if self.config.verbose:
                w = tf.Print(w, [w],
                    message='\nworker w: ', summarize=self.k)

            # calculate policy and sample
            logits = tf.reshape(tf.matmul(self.U, w),[-1, num_acts])

            tf.summary.image('model/logits',
                tf.reshape(logits, (-1, 1, num_acts, 1)), max_outputs=1)

            self.pi = tf.nn.softmax(logits)
            self.log_pi = tf.nn.log_softmax(logits)

            if self.config.verbose:
                self.pi = tf.Print(self.pi, [logits, self.pi, self.log_pi],
                    message='\nworker logits, pi, log_pi: ', summarize=15)

            self.sample = policy_utils.categorical_sample(
                tf.reshape(logits,[-1, num_acts]), num_acts)[0, :]

            if self.config.david_debug:
                self.sample = tf.Print(self.sample,[self.sample,self.pi[0,:],self.prev_g[0,:]],message='Sampled',summarize=6)
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
        if self.config.similarity_metric == 'cosine':
            dot = tf.reduce_sum(tf.multiply(self.s_diff, self.g), axis=1)
            # add epislon in before the norm, otherwise can get nans when
            # g is all zeros and try to backprop the norm
            # can also prevent by stopping gradient, but this actually
            # should be included as part of the backprop
            gcut = self.g + self.config.eps
            # also have to epsilon here as well in case s_diff is 0
            mag = tf.norm(self.s_diff, axis=1) * tf.norm(gcut, axis=1) + self.config.eps
            dcos = dot / mag
            similarity = dcos

            if self.config.verbose:
                dcos = tf.Print(dcos, [dcos, dot, mag],
                    message='\nfeudal dcos, dot, magnitude: ', summarize=15)
            tf.summary.scalar("model/dcos", tf.reduce_mean(dcos))
            tf.summary.scalar("model/dcos_magnitude", tf.reduce_mean(mag))

        elif self.config.similarity_metric == 'gaussian':
            diff = self.s_diff - self.g
            similarity = tf.exp(-tf.reduce_sum(tf.multiply(diff, diff), axis=-1))

            if self.config.verbose:
                similarity = tf.Print(similarity, [similarity],
                    message='\ngaussian pdf of g and s_diff: ', summarize=15)
            tf.summary.scalar("model/gaussian_pdf", tf.reduce_mean(similarity))

        else:
            raise ValueError('invalid similarity metric: {}'.format(
                self.config.similarity_metric))

        cutoff_vf_manager = tf.reshape(tf.stop_gradient(self.manager_vf),[-1])
        if self.config.verbose:
            cutoff_vf_manager = tf.Print(cutoff_vf_manager,
                [self.manager_r, cutoff_vf_manager],
                message='\nfeudal loss return, manager vf: ', summarize=10)

        manager_loss = -tf.reduce_sum((self.manager_r - cutoff_vf_manager) * similarity)

        # manager value loss
        Am = self.manager_r - self.manager_vf
        manager_vf_loss = .5 * tf.reduce_sum(tf.square(Am))

        # worker policy loss
        cutoff_vf_worker = tf.reshape(tf.stop_gradient(self.worker_vf), [-1])
        log_p = tf.reduce_sum(self.log_pi * self.ac, axis=1)
        if self.config.david_debug:
            log_p = tf.Print(log_p,[self.ac[0,:],self.pi[0,:],self.prev_g[0,:]],message='loss',summarize=6)

        if self.config.verbose:
            log_p = tf.Print(log_p, [self.ac],
                message='\naction mask: ', summarize=10)

        alpha = tf.train.polynomial_decay(
            self.config.alpha_start,
            self.global_step,
            end_learning_rate=self.config.alpha_end,
            decay_steps=self.config.alpha_steps,
            power=1
        )

        worker_loss = (self.worker_r + alpha * self.ri - cutoff_vf_worker) * log_p
        worker_loss = -tf.reduce_sum(worker_loss)

        # worker value loss
        Aw = (self.worker_r + alpha * self.ri) - self.worker_vf
        worker_vf_loss = .5 * tf.reduce_sum(tf.square(Aw))

        entropy = -tf.reduce_sum(self.pi * self.log_pi)
        beta = tf.train.polynomial_decay(
            self.config.beta_start,
            self.global_step,
            end_learning_rate=self.config.beta_end,
            decay_steps=self.config.decay_steps,
            power=1
        )

        reg_loss = tf.contrib.layers.apply_regularization(
            tf.contrib.layers.l2_regularizer(self.config.l2_reg),
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                tf.get_variable_scope().name))

        # loss comprised of the individual losses
        self.loss = (
            worker_loss
            + manager_loss
            + worker_vf_loss
            + manager_vf_loss * self.config.manager_value_loss_weight
            - entropy * beta
            + reg_loss
        )

        bs = tf.to_float(tf.shape(self.obs)[0])
        tf.summary.scalar("model/manager_pi_loss", manager_loss / bs)
        tf.summary.scalar("model/worker_pi_loss", worker_loss / bs)
        tf.summary.scalar("model/value_mean", tf.reduce_mean(self.manager_vf))
        tf.summary.scalar("model/manager_value_loss_scaled",
            manager_vf_loss / bs * .5 *  self.config.manager_value_loss_weight)
        tf.summary.scalar("model/worker_value_loss_scaled", worker_vf_loss / bs * .5)
        tf.summary.scalar("model/entropy", entropy / bs)
        tf.summary.scalar("model/mean_log_pi_of_action", tf.reduce_mean(log_p))
        tf.summary.scalar("model/entropy_loss_scaled", -entropy / bs * beta)
        tf.summary.scalar("model/l2_reg_loss_scaled", reg_loss)
        tf.summary.scalar("model/var_global_norm",
            tf.global_norm(tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                tf.get_variable_scope().name)))
        tf.summary.scalar("model/beta", beta)
        tf.summary.scalar("model/alpha", alpha)
        tf.summary.image("model/obs", self.obs, max_outputs=1)

        # additional summaries
        tf.summary.image("model/summed_obs",
            tf.reduce_mean(self.obs, axis=0, keep_dims=True))
        if np.sqrt(self.config.g_dim) == int(np.sqrt(self.config.g_dim)):
            side_length = int(np.sqrt(self.config.g_dim))
            tf.summary.image("model/goal", tf.reshape(
                self.g, (-1, side_length, side_length, 1)), max_outputs=1)
            tf.summary.image("model/s_diff", tf.reshape(
                self.s_diff, (-1, side_length, side_length, 1)), max_outputs=1)
            tf.summary.image("model/goal_mul_s_diff", tf.reshape(
                tf.multiply(self.s_diff, self.g), (-1, side_length, side_length, 1)),
                max_outputs=1)
        tf.summary.scalar("model/intrinsic_reward", tf.reduce_mean(alpha * self.ri))
        tf.summary.scalar("model/manager_return", tf.reduce_mean(self.manager_r))
        tf.summary.scalar("model/worker_return", tf.reduce_mean(self.worker_r))

        self.summary_op = tf.summary.merge_all()

    def get_initial_features(self):
        return np.zeros((1, self.config.c-1, self.g_dim), np.float32),0, self.worker_lstm.state_init + self.manager_state_init

    def act(self, ob, g,idx, cm, hm, cw, hw):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.manager_vf, self.g, self.s, self.last_c_g, self.dilated_idx_out] + self.state_out,
                        {self.obs: [ob], self.state_in[0]: cm, self.state_in[1]: hm,\
                         self.state_in[2]: cw, self.state_in[3]: hw,\
                         self.prev_g: g,self.dilated_idx_in: idx})

    def value(self, ob, g, idx,cm, hm, cw, hw):
        sess = tf.get_default_session()
        manager_vf, worker_vf = sess.run([self.manager_vf, self.worker_vf],
                        {self.obs: [ob], self.state_in[0]: cm, self.state_in[1]: hm,\
                         self.state_in[2]: cw, self.state_in[3]: hw,\
                         self.prev_g: g, self.dilated_idx_in: idx})
        return manager_vf[0], worker_vf[0]

    def update_batch(self, batch):
        return self.batch_processor.process_batch(batch)
