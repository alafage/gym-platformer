import random
from typing import List

from .block import Block
from .chunks import chunks
from .config import Configuration


class Map:
    """ TODO
    """

    def __init__(self, cfg: Configuration) -> None:
        self.cfg = cfg
        self.level = [
            "init",
            "chunk_1",
            "chunk_2",
            "chunk_3",
            "chunk_4",
            "chunk_5",
            "chunk_6",
            "chunk_7",
            "chunk_8",
            "chunk_9",
            "chunk_10",
            "chunk_11",
            "chunk_12",
            "chunk_13",
            "chunk_14",
        ]
        self.blocks: List[Block] = []
        self.level_idx: int = 1
        self.NB_CHUNK = len(self.level)

    def load_chunk(self, key: str, x_start: int) -> None:
        """ TODO
        """
        # gets the chunk
        chunk = chunks[key]
        # sets the x coordinate for the generation.
        x, y = (
            x_start,
            (self.cfg.VISIBILITY_Y - 1)
            * self.cfg.CHUNK_HEIGHT
            * self.cfg.BLOCK_HEIGHT,
        )
        # generation
        for column in range(len(chunk[0])):
            for row in range(len(chunk)):
                if chunk[row][column] == "W":
                    self.blocks.append(Block(x, y, self.cfg))
                elif chunk[row][column] == "E":
                    self.blocks.append(Block(x, y, self.cfg, type="end"))

                y += self.cfg.BLOCK_HEIGHT
            x += self.cfg.BLOCK_WIDTH
            y = (
                (self.cfg.VISIBILITY_Y - 1)
                * self.cfg.CHUNK_HEIGHT
                * self.cfg.BLOCK_HEIGHT
            )

    def end_of_chunk(self) -> bool:
        """ TODO
        """
        return self.blocks[-1].rect.x < self.cfg.SIZE_X

    def level_generation(self) -> bool:
        """ TODO
        """
        if self.end_of_chunk():
            # getting the x coordinate from where to start the generation
            x_start = self.blocks[-1].rect.x + self.cfg.BLOCK_WIDTH

            # random generation
            if self.cfg.RANDOM_GEN:
                # next chunk is chosen randomly
                next_chunk_key = random.choice(list(chunks.keys()))
                self.load_chunk(next_chunk_key, x_start)
                return True
            # sequential generation
            elif self.level_idx < len(self.level):
                # selects the next chunk to be loaded in the chunk list
                next_chunk_key = self.level[self.level_idx]
                self.load_chunk(next_chunk_key, x_start)
                # increments the level index
                self.level_idx += 1
                return True

        return False
