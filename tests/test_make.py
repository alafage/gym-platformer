import numpy as np
from gym import make


def test_make() -> None:
    # Test time boundaries
    env = make("gym_platformer:platformer-v0", ep_duration=10)
    assert not env.observation_space.contains(np.array([0, 0, 0, 0, 10, 0]))
    assert not env.observation_space.contains(np.array([0, 0, 0, 0, -1, 0]))
    assert env.observation_space.contains(np.array([0, 0, 0, 0, 0, 0]))
    assert env.observation_space.contains(np.array([0, 0, 0, 0, 9, 0]))

    env = make("gym_platformer:platformer-v0", ep_duration=100)
    assert not env.observation_space.contains(np.array([0, 0, 0, 0, 100, 0]))
    assert not env.observation_space.contains(np.array([0, 0, 0, 0, -1, 0]))
    assert env.observation_space.contains(np.array([0, 0, 0, 0, 0, 0]))
    assert env.observation_space.contains(np.array([0, 0, 0, 0, 99, 0]))

    env = make("gym_platformer:platformer-v0")
    assert not env.observation_space.contains(np.array([0, 0, 0, 0, 50, 0]))
    assert not env.observation_space.contains(np.array([0, 0, 0, 0, -1, 0]))
    assert env.observation_space.contains(np.array([0, 0, 0, 0, 0, 0]))
    assert env.observation_space.contains(np.array([0, 0, 0, 0, 49, 0]))
