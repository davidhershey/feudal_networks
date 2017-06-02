
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

            n_updates = 100
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
            global_step = tf.get_variable("global_step", [], tf.int32,\
                    initializer=tf.constant_initializer(0, dtype=tf.int32),
                    trainable=False)
            obs_space = (80,80,3)
            act_space = 2
            lr = 5e-4
            g_dim = 256
            worker_hid_dim = 32
            manager_hid_dim = 256
            pi = FeudalPolicy(obs_space, act_space, global_step, test_config.Config())
            
            train_op = tf.train.AdamOptimizer(lr).minimize(pi.loss)

            worker_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            worker_vars = [v for v in worker_vars if 'worker' in v.name]
            worker_assign = tf.group(*[tf.assign(v, tf.zeros_like(v)) 
                for v in worker_vars])

            session.run(tf.global_variables_initializer())

            obs = [np.zeros(obs_space), np.zeros(obs_space)]
            a = [[1,0], [0,1]]
            returns = [0, 1]
            s_diff = [np.ones(g_dim), np.ones(g_dim)]
            gsum = [np.zeros((1,g_dim)), np.ones((1,g_dim))]
            ri = [0, 0]

            _, features = pi.get_initial_features()
            worker_features = features[0:2]
            manager_features = features[2:]

            feed_dict = {
                pi.obs: obs,
                pi.ac: a,
                pi.r: returns,
                pi.s_diff: s_diff,
                pi.prev_g: gsum,
                pi.ri: ri,
                pi.state_in[0]: worker_features[0],
                pi.state_in[1]: worker_features[1],
                pi.state_in[2]: manager_features[0],
                pi.state_in[3]: manager_features[1]
            }

            n_updates = 1000
            verbose = True
            for i in range(n_updates):
                loss, vf, policy, _, _ = session.run(
                    [pi.loss, pi.manager_vf, pi.pi, train_op, worker_assign], 
                    feed_dict=feed_dict)
                
                if verbose:
                    print('loss: {}\npolicy: {}\nvalue: {}\n-------'.format(
                        loss, policy, vf))

                worker_var_values = session.run(worker_vars)
                print(worker_var_values)
                U = session.run(pi.U, feed_dict=feed_dict)
                print(U)
                input()

    

if __name__ == '__main__':
    unittest.main()
