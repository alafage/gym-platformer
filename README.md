# gym-platformer

[![Build Status](https://travis-ci.com/alafage/gym-platformer.svg?branch=master)](https://travis-ci.com/alafage/gym-platformer)
[![codecov](https://codecov.io/gh/alafage/gym-platformer/branch/master/graph/badge.svg)](https://codecov.io/gh/alafage/gym-platformer)

The Platformer environment is a single agent domain featuring continuous state and action spaces. Currently only one map is available (more to come in the future).

## Source

Adaptation of the [simple-platformer](https://github.com/maxenceblanc/simple-platformer) developped by [Maxence Blanc](https://github.com/maxenceblanc) into a `gym` environment.

## Installation

Install the repository with the following command:

```sh
pip install git+https://github.com/alafage/gym-platformer.git
```
It will install `gym` and `pygame` libraries.

## Basic Use

Then you can import the environment into a python file:

```python
import gym

env = gym.make('gym_platformer:platformer-v0')
env.reset()
env.render()
```
