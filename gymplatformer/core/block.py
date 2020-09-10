# fmt: off
import pygame

from .config import Configuration

# fmt: on


class Block:
    """ Block entity
    Parameters
    ----------
    x_coor: int
        Coordinate on x axis.
    y_coor: int
        Coordinate on y axis.
    width: int
        Block width in pixels.
    height: int
        Block height in pixels.
    type: str (default = "default")
        Type of the block.
    """

    def __init__(
        self,
        x_coor: float,
        y_coor: float,
        cfg: Configuration,
        type: str = "default",
    ) -> None:
        # applies coordinates and sizes
        self.rect = pygame.Rect(
            x_coor, y_coor, cfg.BLOCK_WIDTH, cfg.BLOCK_HEIGHT
        )
        # sets block attributes
        self.type = type

    def move(self, x_speed: float, y_speed: float) -> None:
        """ Moves the blocks relatively.
        Parameters
        ----------
        x_speed: float
            Speed on x axis.
        y_speed: float
            Speed on y axis.
        """
        self.rect.x += x_speed
        self.rect.y += y_speed
