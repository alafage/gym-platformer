# gym-platformer

The Platformer environment is a single agent domain featuring continuous state and action spaces.

## Source

Adaptation of the [simple-platformer](https://github.com/maxenceblanc/simple-platformer) developped by [Maxence Blanc](https://github.com/maxenceblanc) into a `gym` environment.

## Installation

```sh
pip install git+https://github.com/alafage/gym-platformer.git
```

Then you can import the environments into a python file by doing:

```python
from gymplatformer import make

env = make("PlatformerEnv")
```
