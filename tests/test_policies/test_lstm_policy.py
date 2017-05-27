import unittest

from feudal_networks.policies.lstm_policy import LSTMPolicy

class config():
    alpha = .5
    size = 256
    n_percept_hidden_layer = 4
    n_percept_filters = 32

class TestLSTMPolicy(unittest.TestCase):

    def test_init(self):
        lstm_pi = LSTMPolicy((80,80,3), 4, config)

if __name__ == '__main__':
    unittest.main()
