

class Policy(object):
    """
    An abstract class defining a learned policy to be used for a Reinforcment
    Learning problem.  This class interfaces with a policy optimizer class
    that oversees training the policy on some environment.

    The policy needs three externally facing methods:
        act()
        value()
        update()
    Which are further documented below.

    Further, upon initialization the following member variables should be
    defined:
        loss        - The tensorflow operation defining the loss function of the
                      policy with respect to a batch of training data
        var_list    - The variables that should be trained by the optimizer
        internals_in- A list of placeholder variables needed at runtime
                      in order to calculate act(), value() or update()
                      (e.g. internal LSTM state)
    """
    def __init__(self,obs_space,act_space,config):
        raise NotImplementedError("Please Implement this method")

    def _build_model(self):
        raise NotImplementedError("Please Implement this method")

    def _build_placeholders(self):
        raise NotImplementedError("Please Implement this method")

    def _build_loss(self):
        """
        Should initialize self.loss to be a tensorflow operation that calculates
        the loss funtion for the current policy
        """
        raise NotImplementedError("Please Implement this method")

    def act(self, obs, prev_internal):
        raise NotImplementedError("Please Implement this method")

    def value(self, obs,prev_internal):
        raise NotImplementedError("Please Implement this method")

    def update(self, sess, train_op, batch):
        raise NotImplementedError("Please Implement this method")
