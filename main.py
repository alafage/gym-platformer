import gymnasium as gym

import gym_platformer  # noqa: F401

if __name__ == "__main__":
    env = gym.make("platformer-v0")
    state, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        env.render()
