import pytest

from gymplatformer import PlatformerEnv
from gymplatformer.utils import custom_score


def test_step() -> None:
    env = PlatformerEnv(custom_score, 10)
    env.reset()
    with pytest.raises(ValueError):
        env.step(50)


def test_reset() -> None:
    env = PlatformerEnv(custom_score, 10)
    env.reset()
    assert hasattr(env, "player")
    assert hasattr(env, "time_ref")
    if hasattr(env, "time_ref"):
        assert isinstance(env.time_ref, float)
    assert env.time_val == 0.0
    assert env.score_val == 0.0
    assert env.completion == 0.0
    assert env.steps_beyond_done is None


def test_render() -> None:
    ...
