from gym.envs.registration import register

register(
    id='platformer-v0',
    entry_point='gym_platformer.envs:PlatformerEnv',
)

register(
    id='discrete-platformer-v0',
    entry_point='gym_platformer.envs:DiscretePlatformerEnv'
)