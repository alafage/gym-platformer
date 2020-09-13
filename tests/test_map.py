from gymplatformer.core import Configuration, Map


def test_valid_chunk() -> None:
    cfg = Configuration(chunk_height=3)
    map = Map(cfg)
    invalid_chunk1 = [
        "  ",
        "  ",
    ]
    assert not map.valid_chunk(invalid_chunk1)
    invalid_chunk2 = [
        " ",
        "  ",
        " ",
    ]
    assert not map.valid_chunk(invalid_chunk2)
    valid_chunk = [
        "  ",
        " W",
        "WW",
    ]
    assert map.valid_chunk(valid_chunk)


def test_load_chunk() -> None:
    cfg = Configuration(chunk_height=3)
    map = Map(cfg)
    chunk_test = [
        " ",
        "W",
        "W",
    ]
    map.load_chunk(chunk_test, 0)
    assert len(map.blocks) == 2


def test_level_generation() -> None:
    cfg = Configuration()
    map = Map(cfg)
    map.load_chunk("init", 0)
    assert map.level_generation()
    assert map.level_generation()
    assert map.level_generation()
    assert not map.level_generation()
