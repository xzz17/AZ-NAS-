import numpy as np

def az_nas_ranking(scores_dict, proxy_ordering=None):
    """
    Perform AZ-NAS ranking to score architectures using multiple proxy metrics.

    Args:
        scores_dict (dict): A dictionary where each key is a proxy metric name
                            and the value is a list of scores for all architectures.
                            Example:
                            {
                                'expressivity': [...],
                                'trainability': [...],
                                ...
                            }

        proxy_ordering (dict or None): Optional dict specifying the ranking direction
            for each proxy metric:
            - 'desc': higher is better
            - 'asc': lower is better
            - 'abs1': closer to 1 is better (e.g., for trainability)
            If None, defaults to 'desc' for all metrics.

    Returns:
        list of float: Final AZ-NAS score for each architecture (lower is better).
                       The score is computed as the sum of the log-normalized ranks
                       across all proxies.
    """
    m = len(next(iter(scores_dict.values())))  # number of architectures
    ranks = {}

    if proxy_ordering is None:
        proxy_ordering = {k: 'desc' for k in scores_dict}

    # Compute ranks for each proxy metric
    for k, scores in scores_dict.items():
        scores = np.array(scores)
        order = proxy_ordering.get(k, 'desc')

        if order == 'asc':
            r = np.argsort(np.argsort(scores)) + 1  # lower is better
        elif order == 'desc':
            r = np.argsort(np.argsort(-scores)) + 1  # higher is better
        elif order == 'abs1':
            r = np.argsort(np.argsort(np.abs(1 - scores))) + 1  # closer to 1 is better
        else:
            raise ValueError(f"Unknown ordering for proxy '{k}': {order}")

        ranks[k] = r

    # Combine all ranks into a final AZ-NAS score using log-normalized sum
    final_scores = [
        sum(np.log(max(ranks[p][i] / m, 1e-8)) for p in ranks)
        for i in range(m)
    ]
    return final_scores
