# gym-platformer

[![Build Status](https://travis-ci.com/alafage/gym-platformer.svg?branch=master)](https://travis-ci.com/alafage/gym-platformer)
[![codecov](https://codecov.io/gh/alafage/gym-platformer/branch/master/graph/badge.svg)](https://codecov.io/gh/alafage/gym-platformer)

The Platformer environment is a single agent domain featuring continuous state and action spaces.

## Source

Adaptation of the [simple-platformer](https://github.com/maxenceblanc/simple-platformer) developped by [Maxence Blanc](https://github.com/maxenceblanc) into a `gym` environment.

## Installation

Either copy the files containing the environments to your own code folder while paying attention to the dependencies, or install the repository with the following command:

```sh
pip install git+https://github.com/alafage/gym-platformer.git
```
## Basic Use

Then you can import the environment into a python file and using it like below:

```python
from gymplatformer import make
import pygame

env = make("PlatformerEnv")

clock = pygame.time.Clock()
# sets 
env.reset()
# game loop
while True:
    clock.tick(15)
    env.render()

    key = pygame.key.get_pressed()

    if key[pygame.K_q]:
        if key[pygame.K_z]:
            env.step(2)
        else:
            env.step(0)
    elif key[pygame.K_d]:
        if key[pygame.K_z]:
            env.step(3)
        else:
            env.step(1)
    elif key[pygame.K_z]:
        env.step(4)
    elif key[pygame.K_ESCAPE]:
        break
    else:
        env.step(5)
```
