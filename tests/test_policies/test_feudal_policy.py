
import numpy as np
import unittest

from feudal_networks.policies.feudal_policy import FeudalPolicy
import tensorflow as tf

class TestFeudalPolicy(unittest.TestCase):

    def setUp(self):
        # reset graph before each test case
        tf.reset_default_graph()

    def test_init(self):
        global_step = tf.get_variable("global_step", [], tf.int32,\
                                        initializer=tf.constant_initializer(0, dtype=tf.int32),
                                        trainable=False)
        feudal = FeudalPolicy((80,80,3), 4, global_step)

    def test_fit_simple_dataset(self):
        with tf.Session() as session:
            global_step = tf.get_variable("global_step", [], tf.int32,\
                                            initializer=tf.constant_initializer(0, dtype=tf.int32),
                                            trainable=False)
            obs_space = (80,80,3)
            act_space = 2
            lr = 1e-3
            g_dim = 256
            worker_hid_dim = 32
            manager_hid_dim = 256
            pi = FeudalPolicy(obs_space, act_space, global_step)
            train_op = tf.train.AdamOptimizer(lr).minimize(pi.loss)
            session.run(tf.global_variables_initializer())

            obs = [np.zeros(obs_space), np.zeros(obs_space)]
            a = [[1,0], [0,1]]
            returns = [0, 1]
            s_diff = [np.zeros(g_dim), np.zeros(g_dim)]
            gsum = [np.zeros((1,g_dim)), np.zeros((1,g_dim))]
            ri = [0, 0]
            worker_features = np.zeros((2, worker_hid_dim))
            manager_features = np.zeros((2, manager_hid_dim))

            feed_dict = {
                pi.obs: obs,
                pi.ac: a,
                pi.r: returns,
                pi.s_diff: s_diff,
                pi.prev_g: gsum,
                pi.ri: ri,
                pi.state_in[0]: worker_features[0].reshape(1,-1),
                pi.state_in[1]: worker_features[1].reshape(1,-1),
                pi.state_in[2]: manager_features[0].reshape(1,-1),
                pi.state_in[3]: manager_features[1].reshape(1,-1)
            }

            n_updates = 10
            for i in range(n_updates):
                loss, _ = session.run([pi.loss, train_op], feed_dict=feed_dict)
                print(loss)

if __name__ == '__main__':
    unittest.main()
