from gym_platformer.utils import compute_reward


def test_compute_reward_chunk_bonus() -> None:
    # Completing a chunk should yield exactly chunk_bonus (no distance change)
    reward = compute_reward(
        chunks_newly_completed=1,
        prev_distance_to_end=100.0,
        curr_distance_to_end=100.0,
        chunk_bonus=1.0,
        distance_weight=0.01,
    )
    assert reward == 1.0


def test_compute_reward_distance_progress() -> None:
    # Getting closer to the end block should yield a positive reward
    reward = compute_reward(
        chunks_newly_completed=0,
        prev_distance_to_end=100.0,
        curr_distance_to_end=80.0,
        chunk_bonus=1.0,
        distance_weight=0.01,
    )
    assert reward > 0.0


def test_compute_reward_moving_away() -> None:
    # Moving away from the end block should yield a negative distance component
    reward = compute_reward(
        chunks_newly_completed=0,
        prev_distance_to_end=80.0,
        curr_distance_to_end=100.0,
        chunk_bonus=1.0,
        distance_weight=0.01,
    )
    assert reward < 0.0


def test_compute_reward_combined() -> None:
    # Chunk completion + distance progress must both contribute positively
    reward_with_chunk = compute_reward(
        chunks_newly_completed=1,
        prev_distance_to_end=100.0,
        curr_distance_to_end=80.0,
        chunk_bonus=1.0,
        distance_weight=0.01,
    )
    reward_no_chunk = compute_reward(
        chunks_newly_completed=0,
        prev_distance_to_end=100.0,
        curr_distance_to_end=80.0,
        chunk_bonus=1.0,
        distance_weight=0.01,
    )
    assert reward_with_chunk > reward_no_chunk


def test_compute_reward_weights() -> None:
    # A higher chunk_bonus must increase the completion component
    low_bonus = compute_reward(
        chunks_newly_completed=1,
        prev_distance_to_end=0.0,
        curr_distance_to_end=0.0,
        chunk_bonus=1.0,
        distance_weight=0.01,
    )
    high_bonus = compute_reward(
        chunks_newly_completed=1,
        prev_distance_to_end=0.0,
        curr_distance_to_end=0.0,
        chunk_bonus=5.0,
        distance_weight=0.01,
    )
    assert high_bonus > low_bonus
