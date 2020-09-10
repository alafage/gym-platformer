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

Then you can import the environments into a python file by doing:

```python
from gymplatformer import make

env = make("PlatformerEnv")
```
