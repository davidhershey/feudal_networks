

class PolicyOptimizer(object):
    def __init__(self, env, global_policy, policy):
        """
        - creates runner
        - builds grads
        - builds train op
        - builds sync op
        """
        pass

    def start(self, sess, summary_writer):
        pass

    def pull_batch_from_queue(self):
        pass

    def train(self, sess):
        """
        This first runs the sync op so that the gradients are computed wrt the 
        current global weights. It then takes a rollout from the runner's queue,
        converts it to a batch, and passes that batch and the train op to the
        policy to perform an update. 
        """
        pass
