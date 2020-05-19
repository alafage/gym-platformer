from gym.envs.registration import register

register(
    id='platformer-v0',
    entry_point='gymplatformer.envs:PlatformerEnv',
)

register(
    id='discrete-platformer-v0',
    entry_point='gymplatformer.envs:DiscretePlatformerEnv'
)