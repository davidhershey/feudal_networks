class Config(object):
    size = 256
    num_local_steps = 80
    n_percept_hidden_layer = 4
    n_percept_filters = 32
    beta_start = .01
    beta_end = .01
    decay_steps = 50000000
    summary_steps = 10
    gamma = .99
    chunks = 4
    lr = .0001
