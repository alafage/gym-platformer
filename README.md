# gym-platformer

The Platformer environment is a single agent domain featuring continuous state and action spaces.

## Source

Adaptation of the [simple-platformer](https://github.com/maxenceblanc/simple-platformer) developped by [Maxence Blanc](https://github.com/maxenceblanc) into a `gym` environment.

## Installation

```sh
cd gym-platformer
pip install -e .
```

Then you can import the environments into a python file:

```python
import gym
import gym_platformer

# Continous platformer environment
continuous_env = gym.make('platformer-v0')
# Discrete platformer environment
discrete_env = gym.make('discrete-platformer-v0')
```
