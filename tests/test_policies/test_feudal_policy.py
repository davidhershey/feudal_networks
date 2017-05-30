import unittest

from feudal_networks.policies.feudal_policy import FeudalPolicy
import tensorflow as tf

class TestFeudalPolicy(unittest.TestCase):

    def test_init(self):
        global_step = tf.get_variable("global_step", [], tf.int32,\
                                        initializer=tf.constant_initializer(0, dtype=tf.int32),
                                        trainable=False)
        feudal = FeudalPolicy((80,80,3), 4,global_step)

if __name__ == '__main__':
    unittest.main()
