"""
Note: adapted from the original debugging environment to have Box obs space

Simple environment with known optimal policy and value function.

This environment has just two actions.
Action 0 yields 0 reward and then terminates the session.
Action 1 yields 1 reward and then terminates the session.

Optimal policy: action 1.

Optimal value function: v(0)=1 (there is only one state, state 0)
"""

import copy
import numpy as np
import gym
from gym import spaces

class OneRoundDeterministicRewardBoxObsEnv(gym.Env):
    def __init__(self, obs_shape=(64,64,1)):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=0, shape=obs_shape)
        self._obs = np.zeros(obs_shape)

    def _step(self, action):
        assert self.action_space.contains(action)
        reward = 1 if action == 1 else 0
        return self._obs, reward, True, {}

    def _reset(self):
        return self._obs

"""
Two rounds 
The first state is all 2s, and you get no reward.
The second state corresponds to the value of the action in the first state.
The reward is choosen from a set of possible rewards based on the sequence of 
actions.

Best action sequence is 0,1 with expected value 3.
"""

class TwoRoundNondeterministicRewardBoxObsEnv(gym.Env):
    def __init__(self, obs_shape=(2,2,1)):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=0, shape=obs_shape)
        self._obs = np.zeros(obs_shape)

    def _get_obs(self):
        if self.firstAction is None:
            self._obs.fill(2)
        else:
            self._obs.fill(self.firstAction)
        return copy.deepcopy(self._obs)


    def _step(self, action):
        rewards = [
            [
                [-1, 1], # expected value 0
                [0, 0, 9] # expected value 3. This is the best path.
            ],
            [
                [0, 2], # expected value 1
                [2, 3] # expected value 2.5
            ]
        ]

        assert self.action_space.contains(action)

        if self.firstAction is None:
            self.firstAction = action
            reward = 0
            done = False
        else:
            reward = np.random.choice(rewards[self.firstAction][action])
            done = True

        return self._get_obs(), reward, done, {}

    def _reset(self):
        self.firstAction = None
        return self._get_obs()