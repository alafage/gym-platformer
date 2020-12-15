def custom_score(time: float, completion: float, x: int) -> float:
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
    # completion score
    comp_s = completion
    # duration score
    time_s = time
    # distance score
    dist_s = x * 1e-4
    # overall score
    score = 5 * (15 * comp_s + 5 * dist_s + 1 * time_s)
    return score
