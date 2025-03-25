import numpy as np

def az_nas_ranking(scores_dict, proxy_ordering=None):
    m = len(next(iter(scores_dict.values())))
    ranks = {}
    if proxy_ordering is None:
        proxy_ordering = {k: 'desc' for k in scores_dict}

    for k, scores in scores_dict.items():
        scores = np.array(scores)
        order = proxy_ordering.get(k, 'desc')
        if order == 'asc':
            r = np.argsort(np.argsort(scores)) + 1
        elif order == 'desc':
            r = np.argsort(np.argsort(-scores)) + 1
        elif order == 'abs1':
            r = np.argsort(np.argsort(np.abs(1 - scores))) + 1
        else:
            raise ValueError(f"Unknown ordering for proxy '{k}': {order}")
        ranks[k] = r

    final_scores = [
        sum(np.log(max(ranks[p][i]/m, 1e-8)) for p in ranks)
        for i in range(m)
    ]
    return final_scores