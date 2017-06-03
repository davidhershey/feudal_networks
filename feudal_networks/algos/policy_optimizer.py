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

def process_rollout(rollout, gamma, lambda_=1.0):
    """
given a rollout, compute its returns and the advantage
"""
    batch_si = np.asarray(rollout.states)
    batch_a = np.asarray(rollout.actions)
    rewards = np.asarray(rollout.rewards)
    vpred_t = np.asarray(rollout.values + [rollout.r])

    rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])
    batch_r = discount(rewards_plus_v, gamma)[:-1]
    delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
    # this formula for the advantage comes "Generalized Advantage Estimation":
    # https://arxiv.org/abs/1506.02438
    batch_adv = discount(delta_t, gamma * lambda_)

    features = rollout.features[0]
    # print features
    return Batch(batch_si, batch_a, batch_adv, batch_r, rollout.terminal, features)

# Batch = namedtuple("Batch", ["obs", "a", "returns", "terminal", "s", "g", "features"])
Batch = namedtuple("Batch", ["si", "a", "adv", "r", "terminal", "features"])

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
        self.r = 0.0
        self.terminal = False
        self.features = []

    def add(self, state, action, reward, value, terminal, features):
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal
        self.features += [features]

    def extend(self, other):
        assert not self.terminal
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.r = other.r
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

def env_runner(env, policy, num_local_steps, summary_writer, visualise):
    """
    The logic of the thread runner.  In brief, it constantly keeps on running
    the policy, and as long as the rollout exceeds a certain length, the thread
    runner appends the policy to the queue.
    """
    last_state = env.reset()
    last_features = policy.get_initial_features()
    length = 0
    rewards = 0

    while True:
        terminal_end = False
        rollout = PartialRollout()

        for _ in range(num_local_steps):
            fetched = policy.act(last_state, *last_features)
            action, value_, features = fetched[0], fetched[1], fetched[2:]
            action_to_take = action.argmax()
            state, reward, terminal, info = env.step(action_to_take)

            # collect the experience
            rollout.add(last_state, action, reward, value_, terminal, last_features)
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
                last_features = policy.get_initial_features()
                print("Episode finished. Sum of rewards: %f. Length: %d" % (rewards, length))
                length = 0
                rewards = 0
                break

        if not terminal_end:
            rollout.r = policy.value(last_state, *last_features)

        # once we have enough experience, yield it, and have the ThreadRunner place it on a queue
        yield rollout

class PolicyOptimizer(object):
    def __init__(self, env, task, policy, config, visualise):
        self.env = env
        self.task = task
        self.config = config

        worker_device = "/job:worker/task:{}/cpu:0".format(task)
        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("global"):
                self.global_step = tf.get_variable("global_step", [], tf.int32,
                    initializer=tf.constant_initializer(0, dtype=tf.int32),
                    trainable=False)
                self.network = LSTMPolicy(
                    env.observation_space.shape, 
                    env.action_space.n,
                    self.global_step,
                    config
                )

        with tf.device(worker_device):
            with tf.variable_scope("local"):
                self.local_network = pi = LSTMPolicy(
                    env.observation_space.shape, 
                    env.action_space.n,
                    self.global_step,
                    config
                )
                
                pi.global_step = self.global_step
            self.policy = pi
            # build runner thread for collecting rollouts
            self.runner = RunnerThread(env, self.policy, 20, visualise)

            # formulate gradients
            grads = tf.gradients(pi.loss, pi.var_list)
            grads, _ = tf.clip_by_global_norm(grads, 40)

            # build sync
            # copy weights from the parameter server to the local model
            self.sync = tf.group(*[v1.assign(v2)
                for v1, v2 in zip(pi.var_list, self.network.var_list)])
            grads_and_vars = list(zip(grads, self.network.var_list))
            # for g,v in grads_and_vars:
            #     print g.name,v.name
            inc_step = self.global_step.assign_add(tf.shape(pi.obs)[0])

            # build train op
            opt = tf.train.AdamOptimizer(1e-4)
            self.train_op = tf.group(opt.apply_gradients(grads_and_vars), inc_step)
            self.summary_writer = None
            self.local_steps = 0

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
        batch = process_rollout(rollout, gamma=self.config.gamma)
        batch = self.policy.update_batch(batch)
        compute_summary = self.task == 0 and self.local_steps % 11 == 0
        should_compute_summary = self.task == 0 and self.local_steps % 11 == 0

        if should_compute_summary:
            fetches = [self.policy.summary_op, self.train_op, self.global_step]
        else:
            fetches = [self.train_op, self.global_step]

        feed_dict = {
            self.policy.obs: batch.si,
            self.network.obs: batch.si,

            self.policy.ac: batch.a,
            self.network.ac: batch.a,

            self.policy.adv: batch.adv,
            self.network.adv: batch.adv,

            self.policy.r: batch.r,
            self.network.r: batch.r,
        }

        for i in range(len(self.policy.state_in)):
            feed_dict[self.policy.state_in[i]] = batch.features[i]
            feed_dict[self.network.state_in[i]] = batch.features[i]

        fetched = sess.run(fetches, feed_dict=feed_dict)

        if should_compute_summary:
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), 
                fetched[-1])
            self.summary_writer.flush()
        self.local_steps += 1
