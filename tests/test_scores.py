from gymplatformer.utils import custom_score


def test_custom_score() -> None:
    stage1_score = custom_score(4.0, 0.2)
    stage2_score = custom_score(20.0, 0.2)
    stage3_score = custom_score(1000.0, 0.3)
    assert stage1_score > stage2_score
    assert stage3_score > stage1_score
