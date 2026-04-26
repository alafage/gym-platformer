def compute_reward(
    chunks_newly_completed: int,
    prev_distance_to_end: float,
    curr_distance_to_end: float,
    chunk_bonus: float = 1.0,
    distance_weight: float = 0.01,
) -> float:
    """Computes the step reward.

    The reward has two components:

    - A fixed bonus for each chunk completed during the step.
    - An incremental reward proportional to the progress made toward the
      nearest upcoming end block (positive when the agent moves closer,
      negative when it moves away).

    Args:
        chunks_newly_completed (int): Number of chunks completed this step.
        prev_distance_to_end (float): Distance to the nearest upcoming end
            block at the previous step (in pixels).
        curr_distance_to_end (float): Distance to the nearest upcoming end
            block at the current step (in pixels).
        chunk_bonus (float): Reward granted per completed chunk.
            Defaults to 1.0.
        distance_weight (float): Scaling factor applied to the distance
            progress component. Defaults to 0.01.

    Returns:
        float: The computed step reward.
    """
    completion_reward = chunk_bonus * chunks_newly_completed
    distance_progress = prev_distance_to_end - curr_distance_to_end
    return completion_reward + distance_weight * distance_progress
