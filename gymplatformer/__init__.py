from gym import make as gym_make

from .platformer_env import PlatformerEnv
from .utils import custom_score


def make(env_name, *make_args, **make_kwargs):
    if env_name == "PlatformerEnv":
        # TODO optimize existence, type and value checks
        ep_duration = 100
        if "ep_duration" in make_kwargs.keys():
            if isinstance(make_kwargs["ep_duration"], int):
                if make_kwargs["ep_duration"] > 0:
                    ep_duration = make_kwargs["ep_duration"]

        return PlatformerEnv(custom_score, ep_duration=ep_duration)
    else:
        return gym_make(env_name, *make_args, **make_kwargs)
