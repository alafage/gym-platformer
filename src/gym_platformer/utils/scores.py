def custom_score(time: float, completion: float, x: int) -> float:
    """Computes the score (between 0 and 100).
    A low play time and a high completion rate are valorized.

    Args:
        time (float): Play time in second.
        completion (float): Completion rate of the map (between 0 and 1).
        x (int): Distance traveled.

    Returns:
        float: The computed score.
    """
    # overall score
    return 5 * ((completion * (10 + 5 * time)) + 5 * x * 1e-4)
