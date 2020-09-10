# fmt: off
from typing import List

import pygame

from .block import Block
from .config import Configuration

# fmt: on


class Player:
    """ Player entity
    """

    def __init__(self, cfg: Configuration) -> None:
        self.cfg = cfg
        self.rect = pygame.Rect(
            self.cfg.START_X,
            self.cfg.START_Y,
            self.cfg.PLAYER_WIDTH,
            self.cfg.PLAYER_HEIGHT,
        )
        self.x_speed: float = 0.0
        self.y_speed: float = 0.0

    def slowdown(self):
        """ Slows the player down.
        """

        if 1 > self.x_speed / self.cfg.SLOWDOWN__X > -1:
            self.x_speed = 0.0

        else:
            self.x_speed = float(int(self.x_speed / self.cfg.SLOWDOWN__X))

    def collisions(
        self, x_speed: float, y_speed: float, blocks: List[Block]
    ) -> None:
        """ Handling of collisions when moving the player.
        Parameters
        ----------
        x_speed: ...
        y_speed: ...
        blocks: ...
        """

        for block in blocks:
            if self.rect.colliderect(block.rect):

                if x_speed > 0:
                    self.rect.right = block.rect.left
                    self.slowdown()
                elif x_speed < 0:
                    self.rect.left = block.rect.right
                    self.slowdown()

                if y_speed > 0:
                    self.rect.bottom = block.rect.top
                    self.y_speed = 0.0
                elif y_speed < 0:
                    self.rect.top = block.rect.bottom
                    self.y_speed = 0.0

    def ground(self, blocks: List[Block]) -> bool:
        """ TODO
        """
        for block in blocks:
            for pixel in range(
                -(self.cfg.BLOCK_WIDTH - 1), self.cfg.BLOCK_WIDTH
            ):
                if (
                    self.rect.bottom == block.rect.top
                    and self.rect.left == block.rect.left + pixel
                ):
                    return True
        return False

    def update_speed(self, action: int, blocks: List[Block]) -> None:
        """ Updates player speed on x and y axis.
        Parameters
        ----------
        action: int
            Action code indicating the next movement of the player.
        blocks: block-list
        """
        # HORIZONTAL MOVEMENTS

        # left
        if action in [0, 2]:
            if self.x_speed > 0:
                self.slowdown()
            else:
                self.x_speed -= self.cfg.ACCELERATION_X
        # right
        elif action in [1, 3]:
            if self.x_speed < 0:
                self.slowdown()
            else:
                self.x_speed += self.cfg.ACCELERATION_X
        else:
            if self.x_speed > 0:
                self.x_speed -= 1.0
            elif self.x_speed < 0:
                self.x_speed += 1.0

        # VERTICAL MOVEMENTS

        # gravity
        if not self.ground(blocks):
            self.y_speed += self.cfg.ACCELERATION_Y

        if action in [2, 3, 4] and self.ground(blocks):
            self.y_speed -= self.cfg.SPEED_Y

        # x speed limit
        if self.x_speed < -self.cfg.SPEED_X:
            self.x_speed = -self.cfg.SPEED_X
        elif self.x_speed > self.cfg.SPEED_X:
            self.x_speed = self.cfg.SPEED_X

    def update_coor(self, blocks: List[Block]) -> None:
        """ Moves the player
        Parameter
        ---------
        blocks: block-list
        """
        self.rect.x += self.x_speed

        # correcting not to get past the middle of the screen
        if self.rect.x > self.cfg.SIZE_X / 2:
            self.rect.x = self.cfg.SIZE_X / 2

        # correcting not to get past the left side of the screen
        if self.rect.x < 0:
            self.rect.x = 0
            self.x_speed = 0.0

        # moves the map when the Player reaches the middle of the screen
        if self.rect.x == self.cfg.SIZE_X / 2 and self.x_speed > 0:
            for block in blocks:
                block.move(-self.x_speed, 0)

        self.collisions(self.x_speed, 0, blocks)

        self.rect.y += self.y_speed
        self.collisions(0, self.y_speed, blocks)

    def step(self, action: int, blocks: List[Block]) -> None:
        """ TODO
        """
        self.update_speed(action, blocks)
        self.update_coor(blocks)
