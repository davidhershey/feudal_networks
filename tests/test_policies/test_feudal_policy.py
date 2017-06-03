
import numpy as np
np.set_printoptions(suppress=True, precision=6)
import sys
import tensorflow as tf
import unittest

sys.path.append('..')

from feudal_networks.policies.feudal_policy import FeudalPolicy
import test_config


class TestFeudalPolicy(unittest.TestCase):

    def setUp(self):
        # reset graph before each test case
        tf.set_random_seed(1)
        np.random.seed(1)
        tf.reset_default_graph()

    def test_init(self):
        global_step = tf.get_variable("global_step", [], tf.int32,\
            initializer=tf.constant_initializer(0, dtype=tf.int32),
            trainable=False)
        feudal = FeudalPolicy((80,80,3), 4, global_step, test_config.Config())

    def test_fit_simple_dataset(self):
        with tf.Session() as session:
            global_step = tf.get_variable("global_step", [], tf.int32,\
                initializer=tf.constant_initializer(0, dtype=tf.int32),
                trainable=False)
            config = test_config.Config()
            obs_space = (80,80,3)
            act_space = 2
            config.learning_rate = 1e-4
            g_dim = 256
            worker_hid_dim = 32
            pi = FeudalPolicy(obs_space, act_space, global_step, config)
            train_op = tf.train.AdamOptimizer(config.learning_rate).minimize(pi.loss)
            session.run(tf.global_variables_initializer())

            _, features = pi.get_initial_features()
            worker_features = features[0:2]
            manager_features = features[2:]

            obs = [np.zeros(obs_space)]
            s_diff = [np.zeros(g_dim)]
            prev_g = [np.zeros((1, g_dim))]

            # single step episode where the agent took action 0 and got return 0
            feed_dict_1 = {
                pi.obs: obs,
                pi.ac: [[1, 0]],
                pi.r: [0],
                pi.s_diff: s_diff,
                pi.prev_g: prev_g,
                pi.ri: [0],
                pi.state_in[0]: worker_features[0],
                pi.state_in[1]: worker_features[1],
                pi.state_in[2]: manager_features[0],
                pi.state_in[3]: manager_features[1]
            }

            # single step episode where the agent took action 1 and got return 1
            feed_dict_2 = {
                pi.obs: obs,
                pi.ac: [[0, 1]],
                pi.r: [1],
                pi.s_diff: s_diff,
                pi.prev_g: prev_g,
                pi.ri: [0],
                pi.state_in[0]: worker_features[0],
                pi.state_in[1]: worker_features[1],
                pi.state_in[2]: manager_features[0],
                pi.state_in[3]: manager_features[1]
            }

            n_updates = 200
            verbose = False
            policy = [0,0]
            vf = 0
            for i in range(n_updates):
                feed_dict = feed_dict_1 if i % 2 == 0 else feed_dict_2
                outputs_list = [pi.loss, pi.manager_vf, pi.pi, train_op]
                loss, vf, policy, _ = session.run(
                    outputs_list, feed_dict=feed_dict)
                if verbose:
                    print('loss: {}\npolicy: {}\nvalue: {}\n-------'.format(
                        loss, policy, vf))
            np.testing.assert_array_almost_equal(policy, [[0,1]])
            self.assertTrue(vf > .4 and vf < .6)

    def test_simple_manager_behavior(self):
        with tf.Session() as session:
            global_step = tf.get_variable("global_step", [], tf.int32,
                    initializer=tf.constant_initializer(0, dtype=tf.int32),
                    trainable=False)
            config = test_config.Config()
            config.g_dim = 2
            config.k = 2
            obs_space = (80,80,3)
            act_space = 2
            lr = 5e-3
            g_dim = config.g_dim
            config.worker_lstm_size = config.g_dim
            pi = FeudalPolicy(obs_space, act_space, global_step, config)

            train_op = tf.train.AdamOptimizer(lr).minimize(pi.loss)

            # assign all worker vars to be zero except for U bias
            worker_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            worker_vars = [v for v in worker_vars
            if ('worker' in v.name and 'phi' not in v.name)]
            worker_zero_assigns = [tf.assign(v, tf.zeros_like(v))
                for v in worker_vars if 'flat_logits/b' not in v.name]

            # assign U_bias so that action 0 is ones and 1 is negative ones
            U_bias = np.ones((act_space, config.k), dtype=np.float32)
            U_bias[1,:] = -1
            U_bias = tf.constant(U_bias.reshape(-1))
            U_bias_var = [v for v in worker_vars if 'flat_logits/b' in v.name][0]
            worker_U_assigns = tf.assign(U_bias_var, U_bias)

            # group them
            worker_assigns = tf.group(
                *worker_zero_assigns,
                worker_U_assigns,
            )

            session.run(tf.global_variables_initializer())

            _, features = pi.get_initial_features()
            worker_features = features[0:2]
            manager_features = features[2:]

            obs = [np.zeros(obs_space)]
            prev_g = [np.zeros((1, g_dim))]

            # single step episode where the agent took action 0 and got return 0
            feed_dict_1 = {
                pi.obs: obs,
                pi.ac: [[1, 0]],
                pi.r: [0],
                pi.prev_g: prev_g,
                pi.ri: [0],
                pi.state_in[0]: worker_features[0],
                pi.state_in[1]: worker_features[1],
                pi.state_in[2]: manager_features[0],
                pi.state_in[3]: manager_features[1]
            }

            # single step episode where the agent took action 1 and got return 1
            feed_dict_2 = {
                pi.obs: obs,
                pi.ac: [[0, 1]],
                pi.r: [1],
                pi.prev_g: prev_g,
                pi.ri: [0],
                pi.state_in[0]: worker_features[0],
                pi.state_in[1]: worker_features[1],
                pi.state_in[2]: manager_features[0],
                pi.state_in[3]: manager_features[1]
            }

            n_updates = 100
            verbose = False
            for i in range(n_updates):
                # set worker weights
                session.run(worker_assigns)

                # run a train update
                feed_dict = feed_dict_1 if i % 2 == 0 else feed_dict_2
                # s_diff needs to be random because the constant cases
                # give poor performance
                feed_dict[pi.s_diff] = [np.random.randn(g_dim)]
                outputs_list = [pi.loss, pi.manager_vf, pi.pi, train_op]
                loss, vf, policy, _ = session.run(
                    outputs_list, feed_dict=feed_dict)
                if verbose:
                    print('loss: {}\npolicy: {}\nvalue: {}\n-------'.format(
                        loss, policy, vf))

            self.assertTrue(policy[0,0] < policy[0,1])
            self.assertTrue(vf > .4 and vf < .6)

    def test_simple_worker_behavior(self):
        with tf.Session() as session:
            global_step = tf.get_variable("global_step", [], tf.int32,
                    initializer=tf.constant_initializer(0, dtype=tf.int32),
                    trainable=False)
            config = test_config.Config()
            config.g_dim = 2
            config.k = 2
            obs_space = (80,80,3)
            act_space = 2
            lr = 5e-3
            g_dim = config.g_dim
            config.worker_lstm_size = config.g_dim
            pi = FeudalPolicy(obs_space, act_space, global_step, config)

            train_op = tf.train.AdamOptimizer(lr).minimize(pi.loss)

            # assign all manager vars to be zero except for bias to g
            manager_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            manager_vars = [v for v in manager_vars if 'manager' in v.name]
            manager_zero_assigns = [tf.assign(v, tf.zeros_like(v))
                for v in manager_vars if 'g_hat/b' not in v.name]

            # assign goal bias so all ones
            g_bias = np.ones((config.g_dim), dtype=np.float32)
            g_bias = tf.constant(g_bias)
            g_bias_var = [v for v in manager_vars if 'g_hat/b' in v.name][0]
            manager_g_assigns = tf.assign(g_bias_var, g_bias)

            # group them
            manager_assigns = tf.group(
                *manager_zero_assigns,
                manager_g_assigns,
            )

            session.run(tf.global_variables_initializer())

            _, features = pi.get_initial_features()
            worker_features = features[0:2]
            manager_features = features[2:]

            obs = [np.zeros(obs_space)]
            prev_g = [np.zeros((1, g_dim))]

            # single step episode where the agent took action 0 and got return 0
            feed_dict_1 = {
                pi.obs: obs,
                pi.ac: [[1, 0]],
                pi.r: [0],
                pi.prev_g: prev_g,
                pi.ri: [0],
                pi.state_in[0]: worker_features[0],
                pi.state_in[1]: worker_features[1],
                pi.state_in[2]: manager_features[0],
                pi.state_in[3]: manager_features[1]
            }

            # single step episode where the agent took action 1 and got return 1
            feed_dict_2 = {
                pi.obs: obs,
                pi.ac: [[0, 1]],
                pi.r: [1],
                pi.prev_g: prev_g,
                pi.ri: [0],
                pi.state_in[0]: worker_features[0],
                pi.state_in[1]: worker_features[1],
                pi.state_in[2]: manager_features[0],
                pi.state_in[3]: manager_features[1]
            }

            n_updates = 200
            verbose = False
            for i in range(n_updates):
                # set worker weights
                session.run(manager_assigns)

                # run a train update
                feed_dict = feed_dict_1 if i % 2 == 0 else feed_dict_2
                # s_diff needs to be random because the constant cases
                # give poor performance
                feed_dict[pi.s_diff] = [np.random.randn(g_dim)]
                outputs_list = [pi.loss, pi.manager_vf, pi.pi, train_op]
                loss, vf, policy, _ = session.run(
                    outputs_list, feed_dict=feed_dict)
                if verbose:
                    print('loss: {}\npolicy: {}\nvalue: {}\n-------'.format(
                        loss, policy, vf))
                    input()

            np.testing.assert_array_almost_equal(policy, [[0,1]])


if __name__ == '__main__':
    unittest.main()
