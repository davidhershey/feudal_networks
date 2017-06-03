class Config(object):
    # common 
    verbose = False
    n_percept_hidden_layer = 4
    n_percept_filters = 32
    beta_start = .01
    beta_end = .005
    decay_steps = 20000000
    summary_steps = 10
    num_local_steps = 20
    
    # feudal
    z_dim = 256
    vf_hidden_size = 64
    eps = 1e-8
    global_norm_clip = 40
    
    # manager
    manager_rnn_type = 'lstm'
    s_dim = 256
    manager_lstm_size = 256
    g_dim = 256
    c = 2 # manager timesteps
    manager_discount = .99
    manager_learning_rate = 5e-5
    s_is_obs = False # skips precept and z, used for visualization mainly
    if s_is_obs:
        # make s_dim and g_dim that of the observation
        s_dim = g_dim = 25

    # worker
    # constrained to match manager_lstm_size because of the way features are stored
    worker_lstm_size = manager_lstm_size
    k = 16 # dimensionality of w
    alpha_start = .01
    alpha_end = 1.
    alpha_steps = decay_steps / 20
    worker_discount = .95
    worker_learning_rate = 1e-4