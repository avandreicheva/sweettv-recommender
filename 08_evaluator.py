"""MAP@5 evaluator: AP normalized by min(5, |R_u|). Users with empty R_u are
excluded from the mean but counted separately."""


def average_precision_at_k(pred_list, relevant_set, k=5):
    """Returns None if relevant_set is empty (user excluded from the mean)."""
    if not relevant_set:
        return None
    hits = 0
    score = 0.0
    for i, item in enumerate(pred_list[:k], start=1):
        if item in relevant_set:
            hits += 1
            score += hits / i
    denom = min(k, len(relevant_set))
    return score / denom


def map_at_k(predictions: dict, ground_truth: dict, k=5):
    """
    predictions:   {user_id: [tv_show_id, ...]}  (ordered, len <= k)
    ground_truth:  {user_id: set(tv_show_id)}     (R_u; may be empty set)
    """
    aps = []
    empty_ru = 0
    for user, rel in ground_truth.items():
        pred = predictions.get(user, [])
        ap = average_precision_at_k(pred, rel, k=k)
        if ap is None:
            empty_ru += 1
            continue
        aps.append(ap)
    n_scored = len(aps)
    return {
        "map@k": sum(aps) / n_scored if n_scored else 0.0,
        "n_scored": n_scored,
        "n_empty_Ru": empty_ru,
        "n_total": len(ground_truth),
    }
