
import numpy as np
from collections import namedtuple
from scipy.stats import multivariate_normal

def cosine_similarity(u, v, eps=1e-8):
    return (np.dot(np.squeeze(u), np.squeeze(v))
            / (np.linalg.norm(u) * np.linalg.norm(v) + eps))

def normalized_gaussian_pdf(x, mean):
    diff = np.squeeze(x - mean)
    return np.exp(-np.dot(diff.T, diff))

Batch = namedtuple("Batch", ["obs", "a", "manager_returns", "worker_returns",
    "s_diff", "ri","g_in", "gsum","g_prev", "idx","features"])

class FeudalBatch(object):
    def __init__(self):
        self.obs = []
        self.g_prev = []
        self.g = []
        self.idx = []
        self.a = []
        self.manager_returns = []
        self.worker_returns = []
        self.s_diff = []
        self.ri = []
        self.gsum = []
        self.features = []

    def add(self, obs, a, manager_returns, worker_returns, s_diff, ri,g, gsum,g_prev,idx, features):
        self.obs += [obs]
        self.g_prev += [g_prev]
        self.g += [g]
        self.idx += [idx]
        self.a += [a]
        self.manager_returns += [manager_returns]
        self.worker_returns += [worker_returns]
        self.s_diff += [s_diff]
        self.ri += [ri]
        self.gsum += [gsum]
        self.features +=[features]
        # if self.features is None:
            # self.features = features

    def get_batch(self):
        batch_obs = np.asarray(self.obs)
        batch_idx = np.asarray(self.idx)
        batch_g_in = np.squeeze(np.asarray(self.g))
        batch_g_prev = np.squeeze(np.asarray(self.g_prev))
        sums = []
        for i in range(len(batch_g_prev)):
            # print batch_g_prev[i].shape
            # s = np.sum(batch_g_prev[i],axis=0) + self.g[i]
            sums.append(np.sum(batch_g_prev[i],axis=0))
            # np.sum(batch_g_prev[i],axis=1)
        # for g in self.g_prev:
        #     print g.shape
        #  self.g_prev
        batch_a = np.asarray(self.a)
        batch_manager_returns = np.asarray(self.manager_returns)
        batch_worker_returns = np.asarray(self.worker_returns)
        batch_sd = np.asarray(self.s_diff)
        if len(batch_sd.shape) >= 2 and batch_sd.shape[1] == 1:
            batch_sd = np.squeeze(batch_sd, axis=1)
        batch_ri = np.asarray(self.ri)
        batch_gs = np.asarray(sums)
        return Batch(batch_obs, batch_a, batch_manager_returns,
            batch_worker_returns, batch_sd, batch_ri, batch_g_in,batch_gs,batch_g_prev,batch_idx[0], self.features[0])

class FeudalBatchProcessor(object):
    """
    This class adapts the batch of PolicyOptimizer to a batch useable by
    the FeudalPolicy.
    """
    def __init__(self, c, pad_method='zeros', similarity_metric='cosine'):
        self.c = c
        self.last_terminal = True
        self.pad_method = pad_method
        if similarity_metric == 'cosine':
            self.similarity = cosine_similarity
        elif similarity_metric == 'gaussian':
            self.similarity = normalized_gaussian_pdf
        else:
            raise ValueError('invalid similarity metric: {}'.format(
                similarity_metric))

    def _pad_front(self, batch):
        if self.pad_method == 'same':
            self.s = [batch.s[0] for _ in range(self.c)]
            self.g = [batch.g[0] for _ in range(self.c)]
        elif self.pad_method == 'zeros': # pad zeros
            self.s = [np.zeros_like(batch.s[0]) for _ in range(self.c)]
            self.g = [np.zeros_like(batch.g[0]) for _ in range(self.c)]
        else:
            raise ValueError('invalid pad method: {}'.format(self.pad_method))

        # prepend with dummy values so indexing is the same
        self.obs = [None for _ in range(self.c)]
        self.g_save = [None for _ in range(self.c)]
        self.g_prev = [None for _ in range(self.c)]
        self.idx = [None for _ in range(self.c)]
        self.a = [None for _ in range(self.c)]
        self.manager_returns = [None for _ in range(self.c)]
        self.worker_returns = [None for _ in range(self.c)]
        self.features = [None for _ in range(self.c)]

    def _pad_back(self, batch):
        if self.pad_method == 'same':
            self.s.extend([batch.s[-1] for _ in range(self.c)])
            self.g.extend([batch.g[-1] for _ in range(self.c)])
        elif self.pad_method == 'zeros': # pad zeros
            self.s.extend([np.zeros_like(batch.s[-1]) for _ in range(self.c)])
            self.g.extend([np.zeros_like(batch.g[-1]) for _ in range(self.c)])
        else:
            raise ValueError('invalid pad method: {}'.format(self.pad_method))

    def _extend(self, batch):
        if self.last_terminal:
            self.last_terminal = False
            self._pad_front(batch)

        # extend with the actual values
        self.obs.extend(batch.obs)
        self.g_save.extend(batch.g)
        self.idx.extend(batch.idx)
        self.g_prev.extend(batch.g_prev)
        self.a.extend(batch.a)
        self.manager_returns.extend(batch.manager_returns)
        self.worker_returns.extend(batch.worker_returns)
        self.s.extend(batch.s)
        self.g.extend(batch.g)
        self.features.extend(batch.features)

        # if this is a terminal batch, then append the final s and g c times
        # note that both this and the above case can occur at the same time
        if batch.terminal:
            self._pad_back(batch)

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
                ri += self.similarity(self.s[t] - self.s[t - i], self.g[t - i])
            ri /= c

            # sum of g values used to derive w, input to the linear transform
            gsum = np.zeros_like(self.g[t - c])
            for i in range(t - c, t):
                gsum += self.g[i]

            # add to the batch
            feudal_batch.add(self.obs[t], self.a[t], self.manager_returns[t],
                self.worker_returns[t], s_diff, ri,self.g_save[t], gsum,self.g_prev[t],self.idx[t], self.features[t])

        # in the terminal case, set reset flag
        if batch.terminal:
            self.last_terminal = True
        # in the general case, forget all but the last 2 * c elements
        # reason being that the first c of those we have already computed
        # a batch for, and the second c need those first c
        else:
            twoc = 2 * self.c
            self.obs = self.obs[-twoc:]
            self.idx = self.idx[-twoc:]
            self.g_save = self.g_save[-twoc:]
            self.g_prev = self.g_prev[-twoc:]
            self.a = self.a[-twoc:]
            self.manager_returns = self.manager_returns[-twoc:]
            self.worker_returns = self.worker_returns[-twoc:]
            self.s = self.s[-twoc:]
            self.g = self.g[-twoc:]
            self.features = self.features[-twoc:]

        return feudal_batch.get_batch()
