import re

import numpy as np
import pytest

from gym_platformer.envs import PlatformerEnv


def test_step() -> None:
    env = PlatformerEnv(ep_duration=10)
    env.reset()
    with pytest.raises(ValueError):
        env.step(50)
    observation, reward, terminated, truncated, info = env.step(5)
    assert isinstance(observation, dict)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    env.time_val = 10
    _, _, terminated, truncated, _ = env.step(5)
    assert truncated is True
    assert terminated is False  # episode ended by time limit, not by completion
    with pytest.warns(
        UserWarning,
        match=re.escape(
            "You are calling 'step()' even though this environment has already returned done = True. You "
            "should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior."
        ),
    ):
        env.step(5)


def test_step_applies_penalty_on_truncation() -> None:
    env_with_penalty = PlatformerEnv(ep_duration=1, truncation_penalty=-2.5, deterministic=True)
    env_without_penalty = PlatformerEnv(ep_duration=1, truncation_penalty=0.0, deterministic=True)

    env_with_penalty.reset(seed=123)
    env_without_penalty.reset(seed=123)

    _, reward_with_penalty, _, truncated, _ = env_with_penalty.step(5)
    _, reward_without_penalty, _, truncated_without_penalty, _ = env_without_penalty.step(5)

    assert truncated is True
    assert truncated_without_penalty is True
    assert (reward_with_penalty - reward_without_penalty) == pytest.approx(-2.5)


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
