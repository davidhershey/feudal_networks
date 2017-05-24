
import policy

class FeudalBatchProcessor(object):
    """
    This class adapts the batch of PolicyOptimizer to a batch useable by 
    the FeudalPolicy. 
    """
    def __init__(self):
        pass

    def process_batch(self, batch):
        """
        Converts a normal batch into one used by the FeudalPolicy update.

        FeudalPolicy requires a batch of the form:

        c previous timesteps - batch size timesteps - c future timesteps

        This class handles the tracking the leading and following timesteps over 
        time. Additionally, it also computes values across timesteps from the 
        batch to provide to FeudalPolicy.
        """
        pass

class FeudalPolicy(policy.Policy):
    """
    Policy of the Feudal network architecture.
    """

    def __init__(self, obs_space, act_space, config):
        pass

    def _build_placeholders(self):
        pass

    def _build_model(self):
        """
        Builds the manager and worker models.
        """
        pass

    def _build_loss(self):
        pass

    def act(self, obs, prev_internal):
        pass

    def value(self, obs, prev_internal):
        pass

    def update(self, sess, train_op, batch):
        """
        This function performs a weight update. It does this by first converting
        the batch into a feudal_batch using a FeudalBatchProcessor. The 
        feudal_batch can then be passed into a session run call directly.
        """
        pass