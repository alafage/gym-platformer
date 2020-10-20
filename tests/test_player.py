from gym_platformer.core import Configuration, Map, Player


def test_slowdown() -> None:
    cfg = Configuration()
    cfg.SLOWDOWN_X = 0.8
    player = Player(cfg)
    player.x_speed = 0.5
    player.slowdown()
    assert player.x_speed == 0.0
    player.x_speed = 2.0
    player.slowdown()
    assert player.x_speed == int(2.0 * 0.8)


def test_step() -> None:
    cfg = Configuration(chunk_height=8)
    cfg.START_X = 2 * cfg.BLOCK_WIDTH
    map = Map(cfg)
    player = Player(cfg)
    chunk_test = [
        "     ",
        "     ",
        "     ",
        "WW WW",
        "W   W",
        "     ",
        "    W",
        "WWWWW",
    ]
    map.load_chunk(chunk_test, 0)
    # test vertical moves
    player.step(4, map.blocks)
    assert player.y_speed == -cfg.SPEED_Y
    assert player.x_speed == 0
    assert player.rect.x == cfg.START_X
    assert player.rect.y == cfg.START_Y - cfg.SPEED_Y
    for _ in range(20):
        player.step(5, map.blocks)
    # test horizontal moves
    player.x_speed = cfg.SPEED_X
    player.step(1, map.blocks)
    assert player.x_speed == cfg.SPEED_X
    assert player.rect.x == 48
    assert player.rect.y == 208
    player.step(1, map.blocks)
    assert player.x_speed < cfg.SPEED_X
    assert player.rect.x == 48
    assert player.rect.y == 208
