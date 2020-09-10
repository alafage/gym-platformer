from gym import make as gym_make

from .platformer_env import PlatformerEnv
from .utils import custom_score


def make(env_name, *make_args, **make_kwargs):
    if env_name == "PlatformerEnv":
        return PlatformerEnv(custom_score)
    else:
        return gym_make(env_name, *make_args, **make_kwargs)
