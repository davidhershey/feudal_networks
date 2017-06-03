
import numpy as np
import unittest

from feudal_networks.policies.feudal_batch_processor import FeudalBatchProcessor, FeudalBatch 

from feudal_networks.algos.feudal_policy_optimizer import Batch

class TestFeudalBatchProcessor(unittest.TestCase):

    def test_simple_c_1(self):
        # simple case ignoring the fact that the different list have 
        # elements with different types 
        c = 1
        fbp = FeudalBatchProcessor(c, pad_method='same')

        obs = [1,2]
        a = [1,2]
        m_returns = [1,2]
        w_returns = [1,2]
        terminal = False
        g = [1,2]
        s = [1,2]
        features = [1,2]
        b = Batch(obs, a, m_returns, w_returns, terminal, g, s, features)
        fb = fbp.process_batch(b)
        np.testing.assert_array_equal(fb.obs, [1])
        np.testing.assert_array_equal(fb.a, [1])
        np.testing.assert_array_equal(fb.manager_returns, [1])
        np.testing.assert_array_equal(fb.worker_returns, [1])
        np.testing.assert_array_equal(fb.s_diff, [1])
        np.testing.assert_array_equal(fb.ri, [0])
        np.testing.assert_array_equal(fb.gsum, [1])
        np.testing.assert_array_equal(fb.features, [1])

        obs = [3,4]
        a = [3,4]
        m_returns = [3,4]
        w_returns = [3,4]
        terminal = False
        g = [3,4]
        s = [3,4]
        features = [3,4]
        b = Batch(obs, a, m_returns, w_returns, terminal, g, s, features)
        fb = fbp.process_batch(b)
        np.testing.assert_array_equal(fb.obs, [2,3])
        np.testing.assert_array_equal(fb.a, [2,3])
        np.testing.assert_array_equal(fb.manager_returns, [2,3])
        np.testing.assert_array_equal(fb.worker_returns, [2,3])
        np.testing.assert_array_equal(fb.s_diff, [1,1])
        self.assertEqual(len(fb.ri), 2)
        np.testing.assert_array_equal(fb.gsum, [1, 2])
        np.testing.assert_array_equal(fb.features, [2])

        obs = [5]
        a = [5]
        m_returns = [5]
        w_returns = [6]
        terminal = True
        g = [5]
        s = [5]
        features = [5]
        b = Batch(obs, a, m_returns, w_returns, terminal, g, s, features)
        fb = fbp.process_batch(b)
        np.testing.assert_array_equal(fb.obs, [4,5])
        np.testing.assert_array_equal(fb.a, [4,5])
        np.testing.assert_array_equal(fb.manager_returns, [4,5])
        np.testing.assert_array_equal(fb.worker_returns, [4,6])
        np.testing.assert_array_equal(fb.s_diff, [1,0])
        self.assertEqual(len(fb.ri), 2)
        np.testing.assert_array_equal(fb.gsum, [3,4])
        np.testing.assert_array_equal(fb.features, [4])

    def test_simple_c_2(self):
        # simple case ignoring the fact that the different list have 
        # elements with different types 
        c = 2
        obs = [1,2]
        a = [1,2]
        m_returns = [1,2]
        w_returns = [1,2]
        terminal = False
        g = [1,2]
        s = [1,2]
        features = [1,2]
        b = Batch(obs, a, m_returns, w_returns, terminal, g, s, features)
        
        fbp = FeudalBatchProcessor(c, pad_method='same')
        fb = fbp.process_batch(b)

        np.testing.assert_array_equal(fb.obs, [])
        np.testing.assert_array_equal(fb.a, [])
        np.testing.assert_array_equal(fb.manager_returns, [])
        np.testing.assert_array_equal(fb.worker_returns, [])
        np.testing.assert_array_equal(fb.s_diff, [])
        np.testing.assert_array_equal(fb.ri, [])
        np.testing.assert_array_equal(fb.gsum, [])
        np.testing.assert_array_equal(fb.features, [])

        obs = [3,4]
        a = [3,4]
        m_returns = [3,4]
        w_returns = [3,4]
        terminal = False
        g = [3,4]
        s = [3,4]
        features = [3,4]
        b = Batch(obs, a, m_returns, w_returns, terminal, g, s, features)
        fb = fbp.process_batch(b)
        np.testing.assert_array_equal(fb.obs, [1,2])
        np.testing.assert_array_equal(fb.a, [1,2])
        np.testing.assert_array_equal(fb.manager_returns, [1,2])
        np.testing.assert_array_equal(fb.worker_returns, [1,2])
        np.testing.assert_array_equal(fb.s_diff, [2,2])
        self.assertEqual(len(fb.ri), 2)
        np.testing.assert_array_equal(fb.gsum, [2,2])
        np.testing.assert_array_equal(fb.features, [1])

        obs = [5]
        a = [5]
        m_returns = [5]
        w_returns = [5]
        terminal = True
        g = [5]
        s = [5]
        features = [5]
        b = Batch(obs, a, m_returns, w_returns, terminal, g, s, features)
        fb = fbp.process_batch(b)
        np.testing.assert_array_equal(fb.obs, [3,4,5])
        np.testing.assert_array_equal(fb.a, [3,4,5])
        np.testing.assert_array_equal(fb.manager_returns, [3,4,5])
        np.testing.assert_array_equal(fb.worker_returns, [3,4,5])
        np.testing.assert_array_equal(fb.s_diff, [2,1,0])
        self.assertEqual(len(fb.ri), 3)
        np.testing.assert_array_equal(fb.gsum, [3,5,7])
        np.testing.assert_array_equal(fb.features, [3])

    def test_simple_terminal_on_start(self):
        c = 2
        fbp = FeudalBatchProcessor(c, pad_method='same')

        obs = [1,2]
        a = [1,2]
        m_returns = [1,2]
        w_returns = [1,2]
        terminal = True
        g = [1,2]
        s = [1,2]
        features = [1,2]
        b = Batch(obs, a, m_returns, w_returns, terminal, g, s, features)
        fb = fbp.process_batch(b)
        np.testing.assert_array_equal(fb.obs, [1,2])
        np.testing.assert_array_equal(fb.a, [1,2])
        np.testing.assert_array_equal(fb.manager_returns, [1,2])
        np.testing.assert_array_equal(fb.worker_returns, [1,2])
        np.testing.assert_array_equal(fb.s_diff, [1,0])
        self.assertEqual(len(fb.ri), 2)
        np.testing.assert_array_equal(fb.gsum, [2,2])
        np.testing.assert_array_equal(fb.features, [1])

    def test_intrinsic_reward_and_gsum_calculation(self):
        c = 2
        fbp = FeudalBatchProcessor(c, pad_method='same')

        obs = a = m_returns = w_returns = features = [None, None, None]
        terminal = True
        s = [np.array([2,1]), np.array([1,2]), np.array([2,3])]
        g = [np.array([1,1]), np.array([2,2]), np.array([3,3])]
        b = Batch(obs, a, m_returns, w_returns, terminal, s, g, features)
        fb = fbp.process_batch(b)
        last_ri = (1. + 1. / np.sqrt(2)) / 2
        np.testing.assert_array_almost_equal(fb.ri, [0,0,last_ri])
        np.testing.assert_array_equal(fb.gsum, 
            [np.array([2,2]), np.array([2,2]), np.array([3,3])])

if __name__ == '__main__':
    unittest.main()
