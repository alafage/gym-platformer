import numpy as np

from gymplatformer import make


def test_make() -> None:
    # Test time boundaries
    env = make("PlatformerEnv", ep_duration=10)
    assert not env.observation_space.contains(np.array([0, 0, 0, 0, 11, 0]))
    assert not env.observation_space.contains(np.array([0, 0, 0, 0, -1, 0]))
    assert env.observation_space.contains(np.array([0, 0, 0, 0, 0, 0]))
    assert env.observation_space.contains(np.array([0, 0, 0, 0, 10, 0]))

    env = make("PlatformerEnv")
    assert not env.observation_space.contains(np.array([0, 0, 0, 0, 101, 0]))
    assert not env.observation_space.contains(np.array([0, 0, 0, 0, -1, 0]))
    assert env.observation_space.contains(np.array([0, 0, 0, 0, 0, 0]))
    assert env.observation_space.contains(np.array([0, 0, 0, 0, 100, 0]))
