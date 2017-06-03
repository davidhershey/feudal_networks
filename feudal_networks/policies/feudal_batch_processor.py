
import numpy as np
from collections import namedtuple

def cosine_similarity(u, v):
    return np.dot(np.squeeze(u),np.squeeze(v)) / (np.linalg.norm(u) * np.linalg.norm(v))

Batch = namedtuple("Batch", ["obs", "a", "returns", "s_diff", "ri", "gsum", "features"])

class FeudalBatch(object):
    def __init__(self):
        self.obs = []
        self.a = []
        self.returns = []
        self.s_diff = []
        self.ri = []
        self.gsum = []
        self.features = None

    def add(self, obs, a, returns, s_diff, ri, gsum, features):
        self.obs += [obs]
        self.a += [a]
        self.returns += [returns]
        self.s_diff += [s_diff]
        self.ri += [ri]
        self.gsum += [gsum]
        if self.features is None:
            self.features = features

    def get_batch(self):
        batch_obs = np.asarray(self.obs)
        batch_a = np.asarray(self.a)
        batch_r = np.asarray(self.returns)
        batch_sd = np.squeeze(np.asarray(self.s_diff))
        batch_ri = np.asarray(self.ri)
        batch_gs = np.asarray(self.gsum)
        return Batch(batch_obs,batch_a,batch_r,batch_sd,batch_ri,batch_gs,self.features)

class FeudalBatchProcessor(object):
    """
    This class adapts the batch of PolicyOptimizer to a batch useable by
    the FeudalPolicy.
    """
    def __init__(self, c):
        self.c = c
        self.last_terminal = True

    def _extend(self, batch):
        if self.last_terminal:
            self.last_terminal = False
            self.s = [batch.s[0] for _ in range(self.c)]
            self.g = [batch.g[0] for _ in range(self.c)]
            # prepend with dummy values so indexing is the same
            self.obs = [None for _ in range(self.c)]
            self.a = [None for _ in range(self.c)]
            self.returns = [None for _ in range(self.c)]
            self.features = [None for _ in range(self.c)]

        # extend with the actual values
        self.obs.extend(batch.obs)
        self.a.extend(batch.a)
        self.returns.extend(batch.returns)
        self.s.extend(batch.s)
        self.g.extend(batch.g)
        self.features.extend(batch.features)

        # if this is a terminal batch, then append the final s and g c times
        # note that both this and the above case can occur at the same time
        if batch.terminal:
            self.s.extend([batch.s[-1] for _ in range(self.c)])
            self.g.extend([batch.g[-1] for _ in range(self.c)])

    def process_batch(self, batch):
        """
        Converts a normal batch into one used by the FeudalPolicy update.

        FeudalPolicy requires a batch of the form:

        c previous timesteps - batch size timesteps - c future timesteps

        This class handles the tracking the leading and following timesteps over
        time. Additionally, it also computes values across timesteps from the
        batch to provide to FeudalPolicy.
        """
        # extend with current batch
        self._extend(batch)

        # unpack and compute bounds
        length = len(self.obs)
        c = self.c

        # normally we cannot compute samples for the last c elements, but
        # in the terminal case, we halluciante values where necessary
        end = length if batch.terminal else length - c

        # collect samples to return in a FeudalBatch
        feudal_batch = FeudalBatch()
        for t in range(c, end):

            # state difference
            s_diff = self.s[t + c] - self.s[t]

            # intrinsic reward
            ri = 0
            # note that this for loop considers s and g values
            # 1 timestep to c timesteps (inclusively) ago
            for i in range(1, c + 1):
                ri_s_diff = self.s[t] - self.s[t - i]
                if np.linalg.norm(ri_s_diff) != 0:
                    ri += cosine_similarity(ri_s_diff, self.g[t - i])
            ri /= c

            # sum of g values used to derive w, input to the linear transform
            gsum = np.zeros_like(self.g[t - c])
            for i in range(t - c, t):
                gsum += self.g[i]

            # add to the batch
            feudal_batch.add(self.obs[t], self.a[t], self.returns[t], s_diff,
                ri, gsum, self.features[t])

        # in the terminal case, set reset flag
        if batch.terminal:
            self.last_terminal = True
        # in the general case, forget all but the last 2 * c elements
        # reason being that the first c of those we have already computed
        # a batch for, and the second c need those first c
        else:
            twoc = 2 * self.c
            self.obs = self.obs[-twoc:]
            self.a = self.a[-twoc:]
            self.returns = self.returns[-twoc:]
            self.s = self.s[-twoc:]
            self.g = self.g[-twoc:]
            self.features = self.features[-twoc:]

        return feudal_batch.get_batch()
