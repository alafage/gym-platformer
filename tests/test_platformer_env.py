import numpy as np
import pytest

from gym_platformer.envs import PlatformerEnv


def test_step() -> None:
    env = PlatformerEnv(ep_duration=10)
    env.reset()
    with pytest.raises(ValueError):
        env.step(50)
    state, reward, done = env.step(5)
    assert isinstance(state, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    env.time_val = 10
    _, _, done = env.step(5)
    assert done is True
    with pytest.warns(UserWarning):
        env.step(5)


def test_reset() -> None:
    env = PlatformerEnv(ep_duration=10)
    env.reset()
    assert hasattr(env, "player")
    assert env.time_val == 0
    assert env.score_val == 0.0
    assert env.completion == 0.0
    assert env.steps_beyond_done is None


def test_render() -> None:
    env = PlatformerEnv(ep_duration=10)
    env.reset()
    view = env.render(mode="rgb_array")
    assert isinstance(view, np.ndarray)
    assert view.shape[0] == env.cfg.SIZE_Y
    assert view.shape[1] == env.cfg.SIZE_X
    assert view.shape[2] == 3
    with pytest.raises(ValueError):
        env.render(mode="random_mode")
