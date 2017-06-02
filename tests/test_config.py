"""
This is a config file used for testing.
The values of this object should not be changed.
If a particular value is needed for a test, then this should be 
loaded, and that particular value should be overwritten.
"""
class Config(object):
    # common 
    verbose = True
    size = 256
    n_percept_hidden_layer = 4
    n_percept_filters = 32
    beta_start = .01
    beta_end = .001
    decay_steps = 50000000
    summary_steps = 10

    # feudal
    z_dim = 256
    vf_hidden_size = 128
    eps = 1e-6

    # manager
    manager_rnn_type = 'lstm'
    g_dim = 256 # s_dim, manager_lstm_size must be this as well
    c = 5 # manager timesteps

    # worker
    # constrained to match g_dim because of the way features are stored
    worker_lstm_size = g_dim 
    k = 16 # dimensionality of w
    alpha = .6