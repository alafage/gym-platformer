def custom_score(time: float, completion: float) -> float:
    """ Computes the score (between 0 and 100).
    Parameters
    ----------
    time: float
        Play time in second.
    completion: float
        Completion rate of the map (between 0 and 1).
    Description
    -----------
        A low play time and a high completion rate are valorized.
    """
    score = (completion * 25) * (2 + 2 * (1 / (1 + time ** 1.5)))
    return round(score, 1)
