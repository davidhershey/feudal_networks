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
