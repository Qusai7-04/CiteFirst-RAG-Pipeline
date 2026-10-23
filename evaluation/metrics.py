"""Document-level ranking metrics. Duplicate ranks consume positions, never credit."""
import math

KS = (1, 3, 5, 10)


def score_ranking(retrieved, gold, k_values=KS, grades=None, judged_ids=None):
    gold = set(gold)
    if not gold:
        raise ValueError("At least one verified relevant item is required")
    if any(type(k) is not int or k < 1 for k in k_values):
        raise ValueError("K must be a positive integer")
    judged = gold if judged_ids is None else set(judged_ids)
    if not gold <= judged:
        raise ValueError("Gold items must be judged")
    result = {}
    first = next((i for i, item in enumerate(retrieved, 1) if item in gold), None)
    result["mrr"] = 1 / first if first else 0.0
    for k in k_values:
        prefix = retrieved[:k]
        hits = len(set(prefix) & gold)
        fully_judged = set(prefix) <= judged
        if fully_judged:
            result[f"p@{k}"] = hits / k  # Missing ranks count as non-relevant.
        result[f"r@{k}"] = hits / len(gold)
        result[f"hit@{k}"] = float(hits > 0)
        if grades is not None and fully_judged:
            if set(grades) != gold or any(type(v) is not int or v < 1 for v in grades.values()):
                raise ValueError("Grades must cover precisely the verified gold items")
            seen = set()
            dcg = 0.0
            for rank, item in enumerate(prefix, 1):
                if item not in seen:
                    dcg += (2 ** grades.get(item, 0) - 1) / math.log2(rank + 1)
                seen.add(item)
            ideal = sum((2 ** grade - 1) / math.log2(rank + 1)
                        for rank, grade in enumerate(sorted(grades.values(), reverse=True)[:k], 1))
            if ideal:
                result[f"ndcg@{k}"] = dcg / ideal
    return result


def macro_average(scored_rows, k_values=KS):
    """Failed/skipped rows must be excluded before calling; count is explicit."""
    rows = list(scored_rows)
    keys = ["mrr"] + [f"{metric}@{k}" for k in k_values
                      for metric in ("p", "r", "hit", "ndcg")]
    return {key: {"mean": sum(row[key] for row in rows if key in row) / count,
                  "count": count}
            for key in keys
            if (count := sum(key in row for row in rows))}
