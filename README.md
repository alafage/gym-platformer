# gym-platformer

The Platformer environment is a single agent domain featuring continuous state and action spaces.

## Source

Adaptation of the [simple-platformer](https://github.com/maxenceblanc/simple-platformer) developped by [Maxence Blanc](https://github.com/maxenceblanc) into a `gym` environment.

## Installation

```sh
cd gym-platformer
pip install -e .
```

Then you can import the environment into a python file:

```python
import gym
import gym_platformer

env = gym.make('platformer-v0')
```
