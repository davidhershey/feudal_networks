
import copy 
import cv2
import gym
from gym import spaces
import numpy as np

class VisionMazeEnv(gym.Env):
    metadata = {
        'render.modes': ['rgb_array'],
    }
    def __init__(self, room_length=3, num_rooms_per_side=2, random_goal=False):
        assert room_length % 2 == 1, "room_length must be odd"
        assert room_length >= 3, "room_length must be greater than 3"
        assert num_rooms_per_side >= 1, "must have at least 1 room"

        self.room_length = room_length
        self.num_rooms_per_side = num_rooms_per_side
        self.random_goal = random_goal
        # 0 = up, 1 = right, 2 = down, 3 = left
        self.action_space = spaces.Discrete(4)
        self.max_pos = room_length * num_rooms_per_side - 1
        if random_goal:
            n_channels = 3
            self.agent_channel = 0
            self.goal_channel = 1
        else:
            n_channels = 1
            self.agent_channel = 0
            self.goal_channel = 0
        n_channels = 3 if random_goal else 1
        obs_space = (self.max_pos + 1, self.max_pos + 1, n_channels)
        self.observation_space = spaces.Box(low=0, high=1, shape=obs_space)
        self.goal_reward = 1
        self._obs = np.zeros(obs_space)
        self._reset()

    def _get_obs(self):
        self._obs.fill(0)
        self._obs[self.state[0], self.state[1], self.agent_channel] = 1
        self._obs[self.goal_state[0], self.goal_state[1], self.goal_channel] = 1
        return copy.deepcopy(self._obs)

    def _reset(self):
        if self.random_goal:
            self.goal_state = [
                np.random.randint(self.max_pos), 
                np.random.randint(self.max_pos)
            ]
        else:
            self.goal_state = [self.max_pos, self.max_pos]
        # start in random state in the maze
        x = np.random.randint(self.max_pos)
        y = np.random.randint(self.max_pos)
        self.state = np.array([x, y])
        return self._get_obs()

    def _step(self, a):
        assert self.action_space.contains(a)
        x, y = self.state

        # up
        if a == 0:
            y = self._step_up(x, y)
        # right
        elif a == 1:
            x = self._step_right(x, y)
        # down
        elif a == 2:
            y = self._step_down(x, y)
        # left
        else:
            x = self._step_left(x, y)

        r, done = 0, False
        if x == self.goal_state[0] and y == self.goal_state[1]:
            r, done = self.goal_reward, True
            
        self.state = np.array([x, y])
        return self._get_obs(), r, done, {'reward': r}

    def _step_up(self, x, y):
        ny = y + 1

        # convert to single room format
        local_ny = ny % self.room_length

        # this condition True indicates passing through wall
        if local_ny == 0:

            # this is only allowed if passing through doorway
            if not (x % self.room_length == self.room_length // 2):
                ny = y

        ny = min(ny, self.max_pos)
        return ny

    def _step_right(self, x, y):
        nx = x + 1

        # convert to single room format
        local_nx = nx % self.room_length

        # this condition True indicates passing through wall
        if local_nx == 0:

            # this is only allowed if passing through doorway
            if not (y % self.room_length == self.room_length // 2):
                nx = x

        nx = min(nx, self.max_pos)
        return nx

    def _step_down(self, x, y):        
        ny = y - 1

        # convert to single room format
        local_ny = ny % self.room_length

        # this condition True indicates passing through wall
        if local_ny == self.room_length - 1:

            # this is only allowed if passing through doorway
            if not (x % self.room_length == self.room_length // 2):
                ny = y

        ny = max(0, ny)
        return ny

    def _step_left(self, x, y):
        nx = x - 1

        # convert to single room format
        local_nx = nx % self.room_length

        # this condition True indicates passing through wall
        if local_nx == self.room_length - 1:

            # this is only allowed if passing through doorway
            if not (y % self.room_length == self.room_length // 2):
                nx = x

        nx = max(0, nx)
        return nx

    def _render(self, mode='rgb_array', close=False):
        frame = self._get_obs()
        if not self.random_goal:
            frame = np.repeat(frame, 3, axis=-1)
            color = np.array([205,50,50]) / 255
            frame[self.goal_state[0], self.goal_state[1], :] = color
        frame *= 255
        frame -= 255
        frame *= -1
        s = self.room_length * self.num_rooms_per_side
        for _ in range(3):
            s *= 2
            frame = cv2.resize(frame, (s, s))
        return np.flip(frame.astype(np.uint8), axis=0)