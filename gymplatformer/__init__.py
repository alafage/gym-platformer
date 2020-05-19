from gym import make as gym_make

from envs.discrete_platformer_env import DiscretePlatformerEnv
from envs.platformer_env import PlatformerEnv

def make(env_name, *make_args, **make_kwargs):
    if env_name == "PlatformerEnv":
        return(PlatformerEnv())
    elif env_name == "DiscretePlatformerEnv":
        return(DiscretePlatformerEnv())
    else:
        return(gym_make(env_name, *make_args, **make_kwargs))
