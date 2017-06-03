
import gym 
import numpy as np
import unittest
import sys
import tensorflow as tf

from feudal_networks.algos.feudal_policy_optimizer import FeudalPolicyOptimizer
from feudal_networks.policies.feudal_policy import FeudalPolicy
import feudal_networks.envs.debug_envs

sys.path.append('..')

import test_config


class TestFeudalPolicyOptimizer(unittest.TestCase):

    def setUp(self):
        # reset graph before each test case
        tf.set_random_seed(1)
        np.random.seed(1)
        tf.reset_default_graph()

    def test_init(self):
        env = gym.make('OneRoundDeterministicRewardBoxObs-v0')
        config = test_config.Config()
        with tf.Session() as session:
            try:
                feudal_opt = FeudalPolicyOptimizer(env, 0, 'feudal', config, False)
            except Exception as e:
                self.fail()

    def test_optimize_one_round_deterministic(self):
        # in the single round, constant state case manager does not do anything
        # really because s_diff is always 0
        env = gym.make('OneRoundDeterministicRewardBoxObs-v0')
        config = test_config.Config()
        config.manager_learning_rate = config.worker_learning_rate = 5e-3
        summary_writer = tf.summary.FileWriter('/tmp/test')
        with tf.Session() as session:
            feudal_opt = FeudalPolicyOptimizer(env, 0, 'feudal', config, False)
            session.run(tf.global_variables_initializer())
            feudal_opt.start(session, summary_writer)
            n_rollout_batches = 100
            for _ in range(n_rollout_batches):
                feudal_opt.train(session)



if __name__ == '__main__':
    unittest.main()
