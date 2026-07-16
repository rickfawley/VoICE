def ranked_actionable_prefixes(ranked_features, feature_names, mask):
    """
    Build actionable prefixes in rank order.

    Example:
      ranked_features = [f3,f1,f2,f4], mask=[1,0,1,1]
      -> [[f3], [f3,f2], [f3,f2,f4]]
    """
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    actionable_ranked = [
        f for f in ranked_features
        if mask[name_to_idx[f]] == 1
    ]

    prefixes = []
    for k in range(1, len(actionable_ranked) + 1):
        subset_names = actionable_ranked[:k]
        subset_idx = [name_to_idx[f] for f in subset_names]
        prefixes.append((subset_names, subset_idx))

    return prefixes