# fmt: off
from typing import Callable, Optional, Tuple

import numpy as np
import pygame
from gym import Env, logger, spaces

from .core import Configuration, Map, Player

# fmt: on


class PlatformerEnv(Env):
    """ PlatformerEnv entity
    Parameter
    ---------
    score_fct: Callable[[float, float, Dict[str, float]], float]
    Description
    -----------
        Continuous platformer environment for reinforcement learning with gym
        library.
    Source
    ------
        TODO
    Observation:
    ------------
        Type: Box(5)
        Num     Observation                     Min         Max
        0       Player Horizontal Position      0           Inf
        1       Player Vertical Position        0           height of the window
        2       Player Horizontal Velocity      -Inf        Inf
        3       Player Vertical Velocity        -Inf        Inf
        4       Time                            0           100
        5       Number of chunk passed          0           number of chunks
    Actions
    -------
        Type: Discrete(6)
        Num     Action
        0       Moves to the left
        1       Moves to the right
        2       Jumps to the left
        3       Jumps to the right
        4       Jumps
        5       Does nothing
    """

    def __init__(self, score_fct: Callable[[float, float], float]) -> None:
        self.cfg = Configuration()
        self.map = Map(self.cfg)
        self.score_fct = score_fct
        self.score_val: float
        self.player: Player
        self.time_ref: float
        self.time_val: float
        self.completion: float
        self.viewer: pygame.Surface
        self.action_space = spaces.Discrete(6)

        low = np.array(
            [0, 0, -np.finfo(np.float32).max, -np.finfo(np.float32).max, 0, 0]
        )
        high = np.array(
            [
                np.finfo(np.float32).max,
                self.cfg.SIZE_Y - self.cfg.PLAYER_HEIGHT,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                100,
                self.map.NB_CHUNK,
            ]
        )
        self.observation_space = spaces.Box(low, high, dtype=np.float64)
        self.steps_beyond_done: int

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """ TODO
        """
        # checks whether the action is valid or not
        if not self.action_space.contains(action):
            raise ValueError(f"{action} ({type(action)}) invalid.")
        # moves the player
        self.player.step(action, self.map.blocks)
        # loads the next chunk if needed
        self.map.level_generation()
        # update time
        self.time_val = pygame.time.get_ticks() / 1000 - self.time_ref
        # get number of chunk passed
        # FIXME: not very viable but works for that list length.
        chunks_passed = 0
        for block in self.map.blocks:
            if block.type == "end" and block.rect.x < self.player.rect.x:
                chunks_passed += 1
        # update state
        state = np.array(
            [
                self.player.rect.x,
                self.player.rect.y,
                self.player.x_speed,
                self.player.y_speed,
                self.time_val,
                chunks_passed,
            ]
        )
        # done
        done = not self.observation_space.contains(state)
        if not done:
            if self.completion != chunks_passed / self.map.NB_CHUNK:
                self.completion = chunks_passed / self.map.NB_CHUNK
                # new score computation
                new_score = self.score_fct(self.time_val, self.completion,)
                # computes action reward
                reward = new_score - self.score_val
                # updates the score
                self.score_val = new_score
            else:
                reward = 0.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return state, reward, done

    def reset(self):
        """ TODO
        """
        self.map.load_chunk("init", self.cfg.START_X)
        self.player = Player(self.cfg)
        self.time_ref = pygame.time.get_ticks() / 1000
        self.time_val = 0.0
        self.score_val = 0.0
        self.completion = 0.0
        self.steps_beyond_done = 0

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """ Generates the environment graphical view.
        Parameter
        ---------
        mode: str
            Available mode:
            - "human" displays a pygame window of the environment.
            - "rgb_array" returns a 3d Numpy Array of the environment (HxWxC).
        """
        # creates the window
        self.viewer = pygame.Surface((self.cfg.SIZE_X, self.cfg.SIZE_Y))
        # draws the background
        self.viewer.fill(self.cfg.GREY)
        # draws each block
        for block in self.map.blocks:
            pygame.draw.rect(self.viewer, self.cfg.WHITE, block.rect)
        # draws the player
        pygame.draw.rect(self.viewer, self.cfg.ORANGE, self.player.rect)
        if mode == "human":
            pygame.init()
            font = pygame.font.Font("freesansbold.ttf", 26)
            text = font.render(
                f"Time: {round(self.time_val, 1)}s | "
                f"Completion: {round(self.completion*100, 0)}% | "
                f"Score: {round(self.score_val, 1)}",
                True,
                (0, 255, 0),
            )
            self.viewer.blit(text, (5, 5))
            screen = pygame.display.set_mode((self.cfg.SIZE_X, self.cfg.SIZE_Y))
            screen.blit(self.viewer, (0, 0))
            # refreshes the window
            pygame.display.update()
            return None
        elif mode == "rgb_array":
            return pygame.surfarray.array3d(self.viewer).swapaxes(0, 1)
        else:
            raise ValueError(
                f"expected 'human' or 'rgb_array' as value for mode argument \
                instead of '{mode}'"
            )

    def close(self) -> None:
        pygame.quit()
