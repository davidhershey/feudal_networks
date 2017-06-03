class Config(object):
    # common 
    verbose = False
    size = 256
    n_percept_hidden_layer = 4
    n_percept_filters = 32
    beta_start = .01
    beta_end = .0001
    decay_steps = 5000000
    summary_steps = 10
    num_local_steps = 20
    learning_rate = 1e-4
    discount = .99

    # feudal
    z_dim = 128
    vf_hidden_size = 64
    eps = 1e-8

    # manager
    manager_rnn_type = 'lstm'
    s_dim = 64
    manager_lstm_size = 64
    g_dim = 64
    c = 5 # manager timesteps

    # worker
    # constrained to match manager_lstm_size because of the way features are stored
    worker_lstm_size = manager_lstm_size
    k = 16 # dimensionality of w
    alpha = .6