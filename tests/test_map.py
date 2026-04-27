import pytest

from gym_platformer.core import Configuration, Map


def test_reset() -> None:
    cfg = Configuration()
    map_obj = Map(cfg)
    # chunk_1 has 11 W blocks + 1 E block = 12 blocks total
    map_obj.load_chunk("chunk_1", 0)
    map_obj.level_idx += 1
    assert len(map_obj.blocks) == 12
    assert map_obj.level_idx == 1
    map_obj.reset()
    # init_chunk has 3 W blocks ("WWW" in the last row of a 3-column chunk)
    assert len(map_obj.blocks) == 3
    assert map_obj.level_idx == 0


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
    # Create a fresh map to test the ValueError (reset() would fail with chunk_height=3
    # because init_chunk has 16 rows)
    map_obj2 = Map(cfg)
    invalid_chunk = [
        " ",
        "W",
        "W",
        "W",  # one row too many for chunk_height=3
    ]
    with pytest.raises(ValueError):
        map_obj2.load_chunk(invalid_chunk, 0)


def test_level_generation() -> None:
    # Use deterministic mode so sequential chunk_1 (12 cols) is always loaded.
    # Geometry: init_chunk last block at x=32, SIZE_X=720.
    # Each chunk_1 extends by 12*16=192 px; takes 4 loads to exceed SIZE_X.
    cfg = Configuration(deterministic=True)
    map_obj = Map(cfg, num_chunks=3)
    map_obj.reset()
    assert map_obj.level_generation()
    assert map_obj.level_generation()
    assert map_obj.level_generation()
    assert not map_obj.level_generation()

    # With random generation the first call should always return True
    cfg.RANDOM_GEN = True
    map_obj = Map(cfg)
    map_obj.reset()
    assert map_obj.level_generation()
