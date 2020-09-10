from gymplatformer.core import Block, Configuration


def test_block() -> None:
    cfg = Configuration()
    block = Block(1, 2, cfg)
    block.move(5.4, 2.1)
    assert block.rect.x == int(1 + 5.4)
    assert block.rect.y == int(2 + 2.1)
