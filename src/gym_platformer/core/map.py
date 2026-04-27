import random

from .block import Block
from .chunks import chunks, init_chunk
from .config import Configuration


class Map:
    _available_chunks = list(chunks.keys())

    def __init__(self, cfg: Configuration, num_chunks: int = 1) -> None:
        self.cfg = cfg
        self.blocks: list[Block] = []
        self.end_blocks: list[Block] = []
        self.level_idx: int = 0
        self.num_chunks = num_chunks

    def reset(self) -> None:
        self.blocks = []
        self.level_idx = 0
        self.load_chunk(init_chunk, self.cfg.START_X)

    def valid_chunk(self, chunk: list[str]) -> bool:
        if len(chunk) == self.cfg.CHUNK_HEIGHT:
            return all(len(chunk[0]) == len(chunk[i]) for i in range(1, len(chunk)))
        return False

    def load_chunk(self, identifier: str | list[str], x_start: int) -> None:

        if isinstance(identifier, str):
            # gets the chunk
            chunk = chunks[identifier]
        elif isinstance(identifier, list):
            if self.valid_chunk(identifier):
                chunk = identifier
            else:
                raise ValueError(
                    "given chunk is invalid."
                    f"The rules are: len(chunk)=={self.cfg.CHUNK_HEIGHT} "
                    f"and the items in the chunk must have th same lenght."
                )
        # sets the x coordinate for the generation.
        x = x_start
        # generation
        for column in range(len(chunk[0])):
            y = (self.cfg.VISIBILITY_Y - 1) * self.cfg.CHUNK_HEIGHT * self.cfg.BLOCK_HEIGHT
            for row in range(len(chunk)):
                if chunk[row][column] == "W":
                    self.blocks.append(Block(x, y, self.cfg))
                elif chunk[row][column] == "E":
                    self.blocks.append(Block(x, y, self.cfg, block_type="end"))
                    self.end_blocks.append(self.blocks[-1])

                y += self.cfg.BLOCK_HEIGHT
            x += self.cfg.BLOCK_WIDTH

    def end_of_chunk(self) -> bool:
        return self.blocks[-1].rect.x < self.cfg.SIZE_X

    def level_generation(self) -> bool:

        if self.end_of_chunk():
            # getting the x coordinate from where to start the generation
            x_start = self.blocks[-1].rect.x + self.cfg.BLOCK_WIDTH

            if self.level_idx < self.num_chunks:
                # random generation
                if self.cfg.RANDOM_GEN:
                    # next chunk is chosen randomly
                    next_chunk_key = random.choice(self._available_chunks)  # noqa: S311
                    self.load_chunk(next_chunk_key, x_start)
                # sequential generation
                else:
                    # selects the next chunk to be loaded in the chunk list
                    next_chunk_key = self._available_chunks[
                        self.level_idx % len(self._available_chunks)
                    ]
                    self.load_chunk(next_chunk_key, x_start)

                self.level_idx += 1
                return True

        return False
