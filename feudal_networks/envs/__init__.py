from gym.envs.registration import register

register(
    id='OneRoundDeterministicRewardBoxObs-v0',
    entry_point='feudal_networks.envs.debug_envs:OneRoundDeterministicRewardBoxObsEnv',
    max_episode_steps=1,
)