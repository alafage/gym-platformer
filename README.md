# gym-platformer

[![codecov](https://codecov.io/gh/alafage/gym-platformer/branch/master/graph/badge.svg)](https://codecov.io/gh/alafage/gym-platformer)

The Platformer environment is a single agent domain featuring continuous state and action spaces. Currently only one map is available (more to come in the future).

## Source

Adaptation of the [simple-platformer](https://github.com/maxenceblanc/simple-platformer) developped by [Maxence Blanc](https://github.com/maxenceblanc) into a [`gymnasium`](https://gymnasium.farama.org/api/env) environment.

## Installation

Make sure you have `uv` installed on your system, then install the repository with the following command:

```sh
git clone https://github.com/alafage/gym-platformer.git
cd gym-platformer
uv sync
```

## Launch the game

You can either launch the game to play it:

```sh
python main.py --game-mode manual
```

Or to watch an AI doing random actions:

```sh
python main.py --game-mode auto
```

## Usage

Then you can import the environment into a python file:

```python
import gymnasium as gym

env = gym.make('gym_platformer:platformer-v0')
```
