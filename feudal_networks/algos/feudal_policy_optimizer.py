"""
Much of the code in this file was originally developed as part of the
universe starter agent: https://github.com/openai/universe-starter-agent
"""
from collections import namedtuple
import numpy as np
import scipy.signal
import tensorflow as tf
import threading
import six.moves.queue as queue

from feudal_networks.policies.lstm_policy import LSTMPolicy
from feudal_networks.policies.feudal_policy import FeudalPolicy

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def process_rollout(rollout, manager_gamma, worker_gamma, lambda_=1.0):
    """
given a rollout, compute its returns and the advantage
"""
    batch_si = rollout.states
    batch_a = rollout.actions

    rewards = rollout.rewards
    manager_rewards_plus_v = rollout.rewards + [rollout.manager_r]
    worker_rewards_plus_v = rollout.rewards + [rollout.worker_r]
    batch_manager_r = discount(manager_rewards_plus_v, manager_gamma)[:-1]
    batch_worker_r = discount(worker_rewards_plus_v, worker_gamma)[:-1]

    batch_s = rollout.ss
    batch_g = rollout.gs
    batch_g_prev = rollout.g_prev

    batch_idx = rollout.idx
    features = rollout.features
    return Batch(batch_si, batch_a, batch_manager_r, batch_worker_r,
        rollout.terminal, batch_s, batch_g,batch_g_prev,batch_idx, features)

Batch = namedtuple("Batch", ["obs", "a", "manager_returns", "worker_returns",
    "terminal", "s", "g","g_prev","idx", "features"])

class PartialRollout(object):
    """
    a piece of a complete rollout.  We run our agent, and process its experience
    once it has processed enough steps.
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.ss = []
        self.gs = []
        self.g_prev = []
        self.idx = []
        self.features = []
        self.manager_r = 0.0
        self.worker_r = 0.0
        self.terminal = False

    def add(self, state, action, reward, value,g,s,g_prev,idx,terminal, features):
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal
        self.features += [features]
        self.gs += [g]
        self.g_prev += [g_prev]
        self.idx += [idx]
        self.ss += [s]

    def extend(self, other):
        assert not self.terminal
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.gs.extend(other.gs)
        self.idx.extend(other.idx)
        self.g_prev.extend(other.g_prev)
        self.ss.extend(other.ss)
        self.manager_r = other.manager_r
        self.worker_r = other.worker_r
        self.terminal = other.terminal
        self.features.extend(other.features)

class RunnerThread(threading.Thread):
    """
    One of the key distinctions between a normal environment and a universe environment
    is that a universe environment is _real time_.  This means that there should be a thread
    that would constantly interact with the environment and tell it what to do.  This thread is here.
    """
    def __init__(self, env, policy, num_local_steps, visualise):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(5)
        self.num_local_steps = num_local_steps
        self.env = env
        self.last_features = None
        self.policy = policy
        self.daemon = True
        self.sess = None
        self.summary_writer = None
        self.visualise = visualise

    def start_runner(self, sess, summary_writer):
        self.sess = sess
        self.summary_writer = summary_writer
        self.start()

    def run(self):
        with self.sess.as_default():
            self._run()

    def _run(self):
        rollout_provider = env_runner(self.env, self.policy, self.num_local_steps, self.summary_writer, self.visualise)
        while True:
            # the timeout variable exists because apparently, if one worker dies, the other workers
            # won't die with it, unless the timeout is set to some large number.  This is an empirical
            # observation.

            self.queue.put(next(rollout_provider), timeout=600.0)

from collections import deque
def env_runner(env, policy, num_local_steps, summary_writer,visualise):
    """
    The logic of the thread runner.  In brief, it constantly keeps on running
    the policy, and as long as the rollout exceeds a certain length, the thread
    runner appends the policy to the queue.
    """
    last_state = env.reset()
    last_c_g,idx,last_features = policy.get_initial_features()
    # print last_c_g
    length = 0
    rewards = 0
    reward_list = deque(maxlen=30)
    outfile = open(summary_writer.get_logdir() + '_rewards.txt','w')
    while True:
        terminal_end = False
        rollout = PartialRollout()

        for local_step_iter in range(num_local_steps):
            # print idx
            fetched = policy.act(last_state,last_c_g,idx, *last_features)
            action, value_, g, s, last_c_g,idx, features = fetched[0], fetched[1], \
                                                    fetched[2], fetched[3], \
                                                    fetched[4], fetched[5][0],\
                                                    fetched[6:]
            action_to_take = action.argmax()

            state, reward, terminal, info = env.step(action_to_take)

            # collect the experience
            rollout.add(last_state, action, reward, value_, g, s,last_c_g,idx, terminal, last_features)
            length += 1
            rewards += reward

            last_state = state
            last_features = features

            if info:
                summary = tf.Summary()
                for k, v in info.items():
                    summary.value.add(tag=k, simple_value=float(v))
                summary_writer.add_summary(summary, policy.global_step.eval())
                summary_writer.flush()

            timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
            if terminal or length >= timestep_limit:
                terminal_end = True
                if length >= timestep_limit or not env.metadata.get('semantics.autoreset'):
                    last_state = env.reset()
                last_c_g,idx,last_features = policy.get_initial_features()
                reward_list.append(rewards)
                mean_reward = np.mean(list(reward_list))
                print("Episode finished. Sum of rewards: %f. Length: %d.  Mean Reward: %f" % (rewards, length,mean_reward))
                outfile.write('{}\t{}\n'.format(policy.global_step.eval(),mean_reward))
                # sleep(.2)
                length = 0
                rewards = 0
                break

        if not terminal_end:
            rollout.manager_r, rollout.worker_r = policy.value(
                last_state, last_c_g,idx, *last_features)

        # once we have enough experience, yield it, and have the ThreadRunner place it on a queue
        yield rollout

class FeudalPolicyOptimizer(object):
    def __init__(self, env, task, policy, config, visualise):
        self.env = env
        self.task = task
        self.config = config

        # when testing only on localhost, use simple worker device
        if config.testing:
            worker_device = "/job:localhost/replica:0/task:0/cpu:0"
            global_device = worker_device
        else:
            worker_device = "/job:worker/task:{}/cpu:0".format(task)
            global_device = tf.train.replica_device_setter(
                1, worker_device=worker_device)

        with tf.device(global_device):
            with tf.variable_scope("global"):
                self.global_step = tf.get_variable("global_step", [], tf.int32,
                    initializer=tf.constant_initializer(0, dtype=tf.int32),
                    trainable=False)
                self.network = FeudalPolicy(
                    env.observation_space.shape,
                    env.action_space.n,
                    self.global_step,
                    config
                )

        with tf.device(worker_device):
            with tf.variable_scope("local"):
                self.local_network = pi = FeudalPolicy(
                    env.observation_space.shape,
                    env.action_space.n,
                    self.global_step,
                    config
                )
                pi.global_step = self.global_step
            self.policy = pi
            # build runner thread for collecting rollouts
            self.runner = RunnerThread(env, self.policy, self.config.num_local_steps, visualise)

            # build sync
            # copy weights from the parameter server to the local model
            self.sync = tf.group(*[v1.assign(v2)
                for v1, v2 in zip(pi.var_list, self.network.var_list)])

            # formulate worker gradients and update
            global_w_vars = [v for v in self.network.var_list if 'worker' in v.name]
            local_w_vars = [v for v in pi.var_list if 'worker' in v.name]
            w_grads = tf.gradients(pi.loss, local_w_vars)
            w_grads = [tf.clip_by_value(grad,-1.,1.) for grad in w_grads]
            #w_grads, _ = tf.clip_by_global_norm(
            #    w_grads, self.config.worker_global_norm_clip)
            w_grads_and_vars = [(g,v) for (g,v) in zip(w_grads, global_w_vars)]
            w_opt = tf.train.RMSPropOptimizer(self.config.worker_learning_rate)
            w_train_op = w_opt.apply_gradients(w_grads_and_vars)

            # formulate manager gradients and update
            global_m_vars = [v for v in self.network.var_list if 'manager' in v.name]
            local_m_vars = [v for v in pi.var_list if 'manager' in v.name]
            m_grads = tf.gradients(pi.loss, local_m_vars)
            m_grads = [tf.clip_by_value(grad,-1.,1.) for grad in m_grads]
            #m_grads, _ = tf.clip_by_global_norm(
            #    m_grads, self.config.manager_global_norm_clip)
            m_grads_and_vars = [(g,v) for (g,v) in zip(m_grads, global_m_vars)]
            m_opt = tf.train.RMSPropOptimizer(self.config.manager_learning_rate)
            self.m_train_op = m_opt.apply_gradients(m_grads_and_vars)

            # combine with global step increment
            self.inc_step = self.global_step.assign_add(tf.shape(pi.obs)[0])
            self.w_train_op = tf.group(w_train_op,self.inc_step)
            self.summary_writer = None
            self.local_steps = 0

            # summaries
            tf.summary.scalar("grads/worker_grad_global_norm", tf.global_norm(w_grads))
            tf.summary.scalar("grads/manager_grad_global_norm", tf.global_norm(m_grads))
            self.summary_op = tf.summary.merge_all()

    def start(self, sess, summary_writer):
        self.runner.start_runner(sess, summary_writer)
        self.summary_writer = summary_writer

    def pull_batch_from_queue(self):
        """
        self explanatory:  take a rollout from the queue of the thread runner.
        """
        rollout = self.runner.queue.get(timeout=600.0)
        while not rollout.terminal:
            try:
                rollout.extend(self.runner.queue.get_nowait())
            except queue.Empty:
                break
        return rollout

    def train(self, sess):
        """
        This first runs the sync op so that the gradients are computed wrt the
        current global weights. It then takes a rollout from the runner's queue,
        converts it to a batch, and passes that batch and the train op to the
        policy to perform an update.
        """
        # copy weights from shared to local
        # this should be run first so that the updates are for the most
        # recent global weights
        sess.run(self.sync)
        rollout = self.pull_batch_from_queue()
        batch = process_rollout(rollout,
            manager_gamma=self.config.manager_discount,
            worker_gamma=self.config.worker_discount)
        batch = self.policy.update_batch(batch)

        should_compute_summary = self.task == 0 and self.local_steps % 11 == 0



        # print 'acting'
        # self.policy.act(batch.obs[0],[batch.g_prev[0]], batch.features[0],\
        #                 batch.features[1],batch.features[2],batch.features[3])

        feed_dict = {
            self.policy.obs: batch.obs,
            self.network.obs: batch.obs,

            self.policy.ac: batch.a,
            self.network.ac: batch.a,

            self.policy.manager_r: batch.manager_returns,
            self.network.manager_r: batch.manager_returns,

            self.policy.worker_r: batch.worker_returns,
            self.network.worker_r: batch.worker_returns,

            self.policy.s_diff: batch.s_diff,
            self.network.s_diff: batch.s_diff,

            self.policy.prev_g: batch.g_prev,
            self.network.prev_g: batch.g_prev,

            self.policy.ri: batch.ri,
            self.network.ri: batch.ri,

            self.policy.dilated_idx_in: batch.idx,
            self.network.dilated_idx_in: batch.idx,

            self.policy.g_in: batch.g_in,
            self.network.g_in: batch.g_in
        }

        for i in range(len(self.policy.state_in)):
            feed_dict[self.policy.state_in[i]] = batch.features[i]
            feed_dict[self.network.state_in[i]] = batch.features[i]


        sess.run(self.m_train_op, feed_dict=feed_dict)

        if should_compute_summary:
            fetches = [self.summary_op, self.policy.summary_op, self.w_train_op,
                self.global_step]
        else:
            fetches = [self.w_train_op, self.global_step]

        fetched = sess.run(fetches, feed_dict=feed_dict)

        if should_compute_summary:
            self.summary_writer.add_summary(
                tf.Summary.FromString(fetched[0]),
                fetched[-1])
            self.summary_writer.add_summary(
                tf.Summary.FromString(fetched[1]),
                fetched[-1])
            self.summary_writer.flush()
        self.local_steps += 1

        return np.mean(batch.manager_returns)
