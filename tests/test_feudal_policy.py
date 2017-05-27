import unittest

from feudal_networks.policies.feudal_policy import FeudalPolicy

class config():
    alpha = .5

class TestFeudalPolicy(unittest.TestCase):

    def test_init(self):
        feudal = FeudalPolicy((80,80,3), 4, config)

if __name__ == '__main__':
    unittest.main()
