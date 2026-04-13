import warnings
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pygame
from gymnasium import Env, spaces

from gym_platformer.core import Configuration, Map, Player
from gym_platformer.utils import custom_score


class PlatformerEnv(Env):
    """PlatformerEnv entity.

    Args:
        score_fct (Callable[..., float]), default=`gym_platformer.utils.custom_score`
            The score function that will be use to compute the overall
            score of the agent.
        ep_duration (float): The duration of the episode in number of environment updates.
            Default to 50.

    Description:
        Continuous platformer environment for reinforcement learning with gym
        and pygame libraries.

    Source:
        Adaptation of the simple-platformer developped by Maxence Blanc into a
        gym environment.
        See https://github.com/maxenceblanc/simple-platformer for more details.

    Observation:
        Type: Box(5)
        Num     Observation                     Min         Max
        0       Game window image (HxWxC)         0         255
        1       Player Horizontal Position        0         Inf
        2       Player Vertical Position          0         Height of the window - Player height
        3       Player Horizontal Velocity     -Inf         Inf
        4       Player Vertical Velocity       -Inf         Inf

    Information:
        Type: Dict
        Num     Information                      Min         Max
        0       Time                              0         Episode duration
        1       Number of chunk passed            0         Number of chunks
        2       Score                             -Inf      Inf

    Actions:
        Type: Discrete(6)
        Num     Action
        0       Moves to the left
        1       Moves to the right
        2       Jumps to the left
        3       Jumps to the right
        4       Jumps
        5       Does nothing
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: Literal["human", "rgb_array"] | None = None,
        score_fct: Callable[..., float] = custom_score,
        ep_duration: float = 50,
    ) -> None:
        self.cfg = Configuration()
        self.map = Map(self.cfg)
        self.score_fct = score_fct
        self.score_val: float
        self.player: Player
        self.time_val: int
        self.ep_duration = ep_duration
        self.completion: float
        self.last_chunk_time: int
        self.viewer: pygame.Surface

        image_shape = (self.cfg.SIZE_Y, self.cfg.SIZE_X, 3)
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(0, 255, shape=image_shape, dtype=np.uint8),
                "player_pos_x": spaces.Box(low=0, high=float("inf"), shape=(1,), dtype=np.float32),
                "player_pos_y": spaces.Box(
                    low=0,
                    high=self.cfg.SIZE_Y - self.cfg.PLAYER_HEIGHT,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "player_vel": spaces.Box(
                    low=-float("inf"), high=float("inf"), shape=(2,), dtype=np.float32
                ),
            }
        )

        self.action_space = spaces.Discrete(6)

        self.steps_beyond_done: int | None

        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _get_obs(self) -> dict[str, Any]:
        return {
            "image": self.render(mode="rgb_array"),
            "player_pos_x": np.array([self.player.rect.x], dtype=np.float32),
            "player_pos_y": np.array([self.player.rect.y], dtype=np.float32),
            "player_vel": np.array([self.player.x_speed, self.player.y_speed], dtype=np.float32),
        }

    def _get_info(self) -> dict[str, Any]:
        return {
            "time": self.time_val,
            "completion": self.completion,
            "score": self.score_val,
        }

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Resets the state of the environment."""
        super().reset(seed=seed)
        self.map.reset()
        self.map.load_chunk("init", self.cfg.START_X)
        self.player = Player(self.cfg)
        self.time_val = 0
        self.score_val = 0.0
        self.completion = 0.0
        self.last_chunk_time = 0
        self.steps_beyond_done = None

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Updates the environment according to an action of the agent.

        Args:
            action (int): A valid action index.

        Returns:
            dict[str, Any]: Observation of the environment.
            float: Reward for making the action.
            bool: Indicates episode completion.
            bool: Indicates episode truncation.
            dict[str, Any]: Additional information about the environment.
        """
        # checks whether the action is valid or not
        if not self.action_space.contains(action):
            raise ValueError(f"{action} ({type(action)}) invalid.")
        # moves the player
        self.player.step(action, self.map.blocks)
        # loads the next chunk if needed
        self.map.level_generation()
        # update time
        self.time_val += 1
        # get number of chunk passed
        # FIXME: not efficient but works for that list length.
        chunks_passed = 0
        for block in self.map.blocks:
            if block.block_type == "end" and block.rect.x < self.player.rect.x:
                chunks_passed += 1

        observation = self._get_obs()
        info = self._get_info()

        done = (
            self.time_val >= self.ep_duration
            or chunks_passed >= self.map.NB_CHUNK
            or self.observation_space.contains(observation) is False
        )

        if not done:
            if self.completion != chunks_passed / self.map.NB_CHUNK:
                self.last_chunk_time = self.time_val
            self.completion = chunks_passed / self.map.NB_CHUNK
            time = 1 - (self.time_val / self.ep_duration)
            # new score computation
            new_score = self.score_fct(time, self.completion, self.player.rect.x)
            # computes action reward
            reward = new_score - self.score_val
            # updates the score
            self.score_val = new_score

        elif self.steps_beyond_done is None:
            # Episode just ended!
            self.steps_beyond_done = 0

            if self.completion != chunks_passed / self.map.NB_CHUNK:
                self.last_chunk_time = self.time_val
            self.completion = chunks_passed / self.map.NB_CHUNK
            time = 1 - (self.last_chunk_time / self.ep_duration)
            # new score computation
            new_score = self.score_fct(time, self.completion, self.player.rect.x)
            # computes action reward
            reward = new_score - self.score_val
            # updates the score
            self.score_val = new_score
        else:
            if self.steps_beyond_done == 0:
                warnings.warn(
                    "You are calling 'step()' even though this environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.",
                    stacklevel=2,
                    category=UserWarning,
                )
            self.steps_beyond_done += 1
            reward = 0.0

        if self.render_mode == "human":
            self.render()

        return observation, reward, done, False, info

    def render(self, mode: str = "human") -> np.ndarray | None:
        """Generates the environment graphical view.

        Args:
            mode (str): Available mode:
                - `"human"` displays a pygame window of the environment.
                - `"rgb_array"` returns a 3d Numpy Array of the environment (HxWxC).
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
            if self.window is None:
                pygame.init()
                screen = pygame.display.set_mode((self.cfg.SIZE_X, self.cfg.SIZE_Y))
                self.window = screen

            if self.clock is None:
                self.clock = pygame.time.Clock()

            # refreshes the window
            font = pygame.font.Font("freesansbold.ttf", 26)
            text = font.render(
                f"Steps: {self.time_val} | "
                f"Completion: {round(self.completion * 100, 0)}% | "
                f"Score: {round(self.score_val, 1)}",
                True,
                (0, 255, 0),
            )
            self.viewer.blit(text, (5, 5))
            self.window.blit(self.viewer, (0, 0))
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])

            return None

        if mode == "rgb_array":
            return pygame.surfarray.array3d(self.viewer).swapaxes(0, 1)

        raise ValueError(
            f"expected 'human' or 'rgb_array' as value for mode argument \
                instead of '{mode}'"
        )

    def close(self) -> None:
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
