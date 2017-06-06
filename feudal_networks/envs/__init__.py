from gym.envs.registration import register

register(
    id='OneRoundDeterministicRewardBoxObs-v0',
    entry_point='feudal_networks.envs.debug_envs:OneRoundDeterministicRewardBoxObsEnv',
    max_episode_steps=1,
    tags = {
        'feudal': True
    }
)

register(
    id='TwoRoundNondeterministicRewardBoxObs-v0',
    entry_point='feudal_networks.envs.debug_envs:TwoRoundNondeterministicRewardBoxObsEnv',
    max_episode_steps=2,
    tags = {
        'feudal': True
    }
)

register(
    id='VisionMaze-v0',
    entry_point='feudal_networks.envs.vision_maze:VisionMazeEnv',
    max_episode_steps=200,
    kwargs = {
        'room_length': 9,
        'num_rooms_per_side': 1
    },
    tags = {
        'feudal': True
    }
)

register(
    id='VisionMaze3-v0',
    entry_point='feudal_networks.envs.vision_maze:VisionMazeEnv',
    max_episode_steps=200,
    kwargs = {
        'room_length': 3,
        'num_rooms_per_side': 1
    },
    tags = {
        'feudal': True
    }
)

register(
    id='VisionMaze5-v0',
    entry_point='feudal_networks.envs.vision_maze:VisionMazeEnv',
    max_episode_steps=200,
    kwargs = {
        'room_length': 5,
        'num_rooms_per_side': 1
    },
    tags = {
        'feudal': True
    }
)

register(
    id='VisionMaze9-v0',
    entry_point='feudal_networks.envs.vision_maze:VisionMazeEnv',
    max_episode_steps=200,
    kwargs = {
        'room_length': 9,
        'num_rooms_per_side': 1
    },
    tags = {
        'feudal': True
    }
)

register(
    id='VisionMaze25-v0',
    entry_point='feudal_networks.envs.vision_maze:VisionMazeEnv',
    max_episode_steps=200,
    kwargs = {
        'room_length': 9,
        'num_rooms_per_side': 1
    },
    tags = {
        'feudal': True
    }
)

register(
    id='RoomMaze3-v0',
    entry_point='feudal_networks.envs.vision_maze:VisionMazeEnv',
    max_episode_steps=200,
    kwargs = {
        'room_length': 6,
        'num_rooms_per_side': 3
    },
    tags = {
        'feudal': True
    }
)

register(
    id='RoomMaze5-v0',
    entry_point='feudal_networks.envs.vision_maze:VisionMazeEnv',
    max_episode_steps=200,
    kwargs = {
        'room_length': 6,
        'num_rooms_per_side': 5
    },
    tags = {
        'feudal': True
    }
)

register(
    id='RoomMaze7-v0',
    entry_point='feudal_networks.envs.vision_maze:VisionMazeEnv',
    max_episode_steps=200,
    kwargs = {
        'room_length': 6,
        'num_rooms_per_side': 7
    },
    tags = {
        'feudal': True
    }
)

register(
    id='RoomMaze9-v0',
    entry_point='feudal_networks.envs.vision_maze:VisionMazeEnv',
    max_episode_steps=200,
    kwargs = {
        'room_length': 6,
        'num_rooms_per_side': 9
    },
    tags = {
        'feudal': True
    }
)

register(
    id='RoomMaze-v0',
    entry_point='feudal_networks.envs.vision_maze:VisionMazeEnv',
    max_episode_steps=200,
    kwargs = {
        'room_length': 6,
        'num_rooms_per_side': 9
    },
    tags = {
        'feudal': True
    }
)

register(
    id='RandomGoalVisionMaze-v0',
    entry_point='feudal_networks.envs.vision_maze:VisionMazeEnv',
    max_episode_steps=200,
    kwargs = {
        'room_length': 9,
        'num_rooms_per_side': 1,
        'random_goal': True
    },
    tags = {
        'feudal': True
    }
)
