"""
This is a config file used for testing.
The values of this object should not be changed.
If a particular value is needed for a test, then this should be 
loaded, and that particular value should be overwritten.
"""
class Config(object):
    # common 
    verbose = True
    testing = True
    n_percept_hidden_layer = 4
    n_percept_filters = 32
    beta_start = .01
    beta_end = .001
    decay_steps = 50000000
    summary_steps = 10
    num_local_steps = 40
    global_norm_clip = 40

    # lstm 
    size = 64

    # feudal
    z_dim = 64
    vf_hidden_size = 128
    eps = 1e-6
    batch_pad_method = 'same' # tests assume same, but zeros performs better
    
    # manager
    manager_rnn_type = 'lstm'
    s_dim = 64
    manager_lstm_size = 64
    g_dim = 64 # s_dim, manager_lstm_size must be this as well
    c = 5 # manager timesteps
    s_is_obs = False # skips precept and z, used for visualization mainly
    manager_learning_rate = 1e-4
    manager_discount = .99
    
    # worker
    # constrained to match g_dim because of the way features are stored
    worker_lstm_size = g_dim 
    k = 16 # dimensionality of w
    worker_learning_rate = 1e-4
    worker_discount = .99
    alpha_start = .5
    alpha_end = .5
    alpha_steps = decay_steps / 20