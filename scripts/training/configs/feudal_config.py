class Config(object):
    # common 
    verbose = False
    n_percept_hidden_layer = 4
    n_percept_filters = 32
    beta_start = .05
    beta_end = .05
    decay_steps = 1000000
    summary_steps = 10
    num_local_steps = 20
    testing = False
    l2_reg = 0.
    dropout_keep_prob = .5
    
    # feudal
    z_dim = 64
    vf_hidden_size = 64
    eps = 1e-8
    # how the batch is padded at the beginning and end of the episode
    # zeros typically will do better when the critical events occur at the 
    # beginning and end of the episode because they yield absolute direction
    # goals rather than no goals, which is what same yields.
    batch_pad_method = 'same' # 'same'
    similarity_metric = 'gaussian' # 'gaussian'
    
    # manager
    manager_rnn_type = 'lstm'
    s_dim = 64
    manager_lstm_size = 64
    g_dim = 64
    c = 8 # manager timesteps
    manager_global_norm_clip = 40
    manager_discount = .999
    manager_learning_rate = 5e-4
    manager_value_loss_weight = .1
    s_is_obs = False # skips precept and z, used for visualization mainly
    if s_is_obs:
        # make s_dim and g_dim that of the observation
        s_dim = g_dim = 25

    # worker
    # constrained to match manager_lstm_size because of the way features are stored
    worker_lstm_size = manager_lstm_size
    k = 16 # dimensionality of w
    alpha_start = .5
    alpha_end = .5
    alpha_steps = decay_steps / 20
    worker_discount = .95
    worker_learning_rate = 5e-4
    worker_global_norm_clip = 40
    worker_hint = True