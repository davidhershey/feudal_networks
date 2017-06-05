class Config(object):
    # common
    verbose = False
    n_percept_hidden_layer = 4
    n_percept_filters = 32
    beta_start = .001
    beta_end = .001
    decay_steps = 5000000
    summary_steps = 10
    num_local_steps = 400
    testing = False
    l2_reg = 0.0
    dropout_keep_prob = 1.0
    use_batch_norm = False

    # feudal
    z_dim = 64
    vf_hidden_size = 64
    eps = 1e-8
    # how the batch is padded at the beginning and end of the episode
    # zeros typically will do better when the critical events occur at the
    # beginning and end of the episode because they yield absolute direction
    # goals rather than no goals, which is what same yields.
    batch_pad_method = 'zeros' # 'same'
    similarity_metric = 'cosine' # 'gaussian'

    # manager
    manager_rnn_type = 'dilated'
    s_dim = 256
    manager_lstm_size = 256
    g_dim = 256
    c = 8 # manager timesteps
    manager_global_norm_clip = 100
    manager_discount = .99
    manager_learning_rate = 1e-4
    manager_value_loss_weight = .1

    g_eps_start = .10
    g_eps_end = 0.0
    g_eps_steps = 10000000
    # g_eps = .02 # Probability of random goal
    s_is_obs = False # skips precept and z, used for visualization mainly
    if s_is_obs:
        # make s_dim and g_dim that of the observation
        s_dim = g_dim = 25

    david_debug = False

    # worker
    # constrained to match manager_lstm_size because of the way features are stored
    worker_lstm_size = manager_lstm_size
    k = 16 # dimensionality of w
    alpha_start = .35
    alpha_end = .35
    alpha_steps = decay_steps / 20
    worker_discount = .95
    worker_learning_rate = 1e-4
    worker_global_norm_clip = 40
    worker_hint = False
