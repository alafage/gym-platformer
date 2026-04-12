import pytest

from gym_platformer.core import Configuration, Map


def test_reset() -> None:
    cfg = Configuration(chunk_height=3)
    map_obj = Map(cfg)
    chunk = [" ", " ", "W"]
    map_obj.load_chunk(chunk, 0)
    map_obj.level_idx += 1
    assert len(map_obj.blocks) == 1
    assert map_obj.level_idx == 2
    map_obj.reset()
    assert len(map_obj.blocks) == 0
    assert map_obj.level_idx == 1


def test_valid_chunk() -> None:
    cfg = Configuration(chunk_height=3)
    map_obj = Map(cfg)
    invalid_chunk1 = [
        "  ",
        "  ",
    ]
    assert not map_obj.valid_chunk(invalid_chunk1)
    invalid_chunk2 = [
        " ",
        "  ",
        " ",
    ]
    assert not map_obj.valid_chunk(invalid_chunk2)
    valid_chunk = [
        "  ",
        " W",
        "WW",
    ]
    assert map_obj.valid_chunk(valid_chunk)


def test_load_chunk() -> None:
    cfg = Configuration(chunk_height=3)
    map_obj = Map(cfg)
    chunk_test = [
        " ",
        "W",
        "W",
    ]
    map_obj.load_chunk(chunk_test, 0)
    assert len(map_obj.blocks) == 2
    map_obj.reset()
    chunk_test.append("W")
    with pytest.raises(ValueError):
        map_obj.load_chunk(chunk_test, 0)


def test_level_generation() -> None:
    cfg = Configuration()
    map_obj = Map(cfg)
    map_obj.load_chunk("init", 0)
    assert map_obj.level_generation()
    assert map_obj.level_generation()
    assert map_obj.level_generation()
    assert not map_obj.level_generation()

    cfg.RANDOM_GEN = True
    map_obj = Map(cfg)
    map_obj.load_chunk("init", 0)
    assert map_obj.level_generation()
