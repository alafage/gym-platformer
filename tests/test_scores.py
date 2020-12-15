from gym_platformer.utils import custom_score


def test_custom_score() -> None:
    stage1_score = custom_score(0.9, 0.2, 0)
    stage2_score = custom_score(0.8, 0.2, 0)
    stage3_score = custom_score(0.0, 0.3, 0)
    stage4_score = custom_score(0.9, 0.2, 1)

    assert stage1_score > stage2_score
    assert stage3_score > stage1_score
    assert stage4_score > stage1_score
