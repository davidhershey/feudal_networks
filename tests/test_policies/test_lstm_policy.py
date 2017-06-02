
import sys
import unittest
import tensorflow as tf

sys.path.append('..')

from feudal_networks.policies.lstm_policy import LSTMPolicy
import test_config


class TestLSTMPolicy(unittest.TestCase):

    def test_init(self):
        global_step = tf.get_variable("global_step", [], tf.int32,\
                                        initializer=tf.constant_initializer(0, dtype=tf.int32),
                                        trainable=False)
        try:
            lstm_pi = LSTMPolicy((80,80,3), 4, global_step, test_config.Config())
        except Exception as e:
            self.fail('raised exception: {}'.format(e))

if __name__ == '__main__':
    unittest.main()
