
import gym
import numpy as np
import tensorflow as tf

from feudal_networks.models.models import (linear, conv2d, build_lstm,
    normalized_columns_initializer)
import feudal_networks.policies.policy_utils as policy_utils

from configs.lstm_config import Config

class LSTMPolicy(object):
    def __init__(self, obs_space, act_space):
        self.obs_space = obs_space
        self.act_space = act_space
        self.config = config = Config
        self.local_steps = 0
        # build placeholders
        self.obs = x = tf.placeholder(tf.float32, 
                                    [None] + list(obs_space), 
                                    name='state')
        self.adv = tf.placeholder(tf.float32, 
                                    [None], 
                                    name="adv")
        self.ac = tf.placeholder(tf.float32, 
                                    [None, act_space], 
                                    name="ac")
        self.r = tf.placeholder(tf.float32, 
                                    [None], 
                                    name="r")

        # build perception
        for i in range(config.n_percept_hidden_layer):
            x = tf.nn.elu(conv2d(x, config.n_percept_filters, 
                "l{}".format(i + 1), [3, 3], [2, 2]))

        # introduce a "fake" batch dimension of 1 after flatten so that we 
        # can do LSTM over time dim
        x = tf.expand_dims(policy_utils.flatten(x), [0])
        x, self.state_init, self.state_in, self.state_out = build_lstm(
            x, config.size, 'lstm', tf.shape(self.x)[:1])

        # add hidden layer to value output so that less of a burden is placed 
        # on the lstm to output values for both the policy and value function
        vfhid = tf.nn.elu(linear(x, config.size, "value_hidden", 
            normalized_columns_initializer(0.01)))
        self.vf = tf.reshape(linear(vfhid, 1, "value", 
            normalized_columns_initializer(1.0)), [-1])

        # retrieve logits, sampling op
        self.logits = linear(x, act_space, "action", 
            normalized_columns_initializer(0.01))
        self.sample = policy_utils.categorical_sample(
            self.logits, act_space)[0, :]
        self.var_list = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

        # build loss
        log_prob_tf = tf.nn.log_softmax(self.logits)
        prob_tf = tf.nn.softmax(self.logits)
        pi_loss = - tf.reduce_sum(
                    tf.reduce_sum(log_prob_tf * self.ac, [1]) * self.adv)
        entropy = - tf.reduce_sum(prob_tf * log_prob_tf)
        vf_loss = 0.5 * tf.reduce_sum(tf.square(self.vf - self.r))
        beta = tf.train.polynomial_decay(config.beta_start, self.global_step, 
                end_learning_rate=config.beta_end,
                decay_steps=config.decay_steps, 
                power=1)
        self.loss = pi_loss + 0.5 * vf_loss - entropy * beta

        # summaries
        bs = tf.to_float(tf.shape(self.x)[0])
        tf.summary.scalar("model/policy_loss", pi_loss / bs)
        tf.summary.scalar("model/value_mean", tf.reduce_mean(self.vf))
        tf.summary.scalar("model/value_loss", vf_loss / bs)
        tf.summary.scalar("model/value_loss_scaled", vf_loss / bs * .5)
        tf.summary.scalar("model/entropy", entropy / bs)
        tf.summary.scalar("model/entropy_loss_scaleed", -entropy / bs * beta)
        tf.summary.scalar("model/grad_global_norm", tf.global_norm(grads))
        tf.summary.scalar("model/var_global_norm", tf.global_norm(self.var_list))
        tf.summary.scalar("model/beta", beta)
        tf.summary.image("model/state", self.x)
        self.summary_op = tf.summary.merge_all()

    def get_initial_features(self):
        return self.state_init

    def act(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf] + self.state_out,
                        {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})

    def value(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})[0]

    def update(self, sess, train_op, batch, global_step, summary_writer):
        self.local_steps += 1
        compute_summary = self.local_steps % self.config.summary_steps == 0
        if compute_summary:
            fetches = [self.summary_op, train_op, global_step]
        else:
            fetches = [train_op, global_step]

        feed_dict = {
            self.x: batch.si,
            self.ac: batch.a,
            self.adv: batch.adv,
            self.r: batch.r,
            self.state_in[0]: batch.features[0],
            self.state_in[1]: batch.features[1],
        }

        fetched = sess.run(fetches, feed_dict=feed_dict)

        if compute_summary:
            summary_writer.add_summary(
                tf.Summary.FromString(fetched[0]), fetched[-1])
            summary_writer.flush()
        
