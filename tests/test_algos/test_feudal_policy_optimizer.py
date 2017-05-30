
import gym 
import unittest
import tensorflow as tf

from feudal_networks.algos.feudal_policy_optimizer import FeudalPolicyOptimizer
from feudal_networks.policies.feudal_policy import FeudalPolicy

import feudal_networks.envs.debug_envs

class TestFeudalPolicyOptimizer(unittest.TestCase):

    def test_init(self):
        env = gym.make('OneRoundDeterministicRewardBoxObs-v0')
        with tf.Session() as session:
            feudal_opt = FeudalPolicyOptimizer(env, 0, 'feudal', False)

if __name__ == '__main__':
    unittest.main()
