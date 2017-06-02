class Config(object):
    
    alpha = .6
    vf_hidden_size = 128
    k = 16 # Dimensionality of w
    g_dim = 600
    c = 5
    beta_start = .01
    beta_end = .00001
    decay_steps = 500000
    worker_lstm_size = 256

    z_dim = 256
    rnn_type = 'lstm'
