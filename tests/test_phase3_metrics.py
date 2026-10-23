import pytest

from evaluation.metrics import macro_average, score_ranking


def test_perfect_ranking_and_multiple_gold():
    scores = score_ranking(["A", "B"], {"A", "B"}, (1, 3))
    assert scores["p@1"] == 1
    assert scores["r@1"] == .5
    assert scores["r@3"] == 1
    assert scores["p@3"] == 2 / 3
    assert scores["mrr"] == 1


def test_rank_k_missing_results_and_duplicates():
    scores = score_ranking(["X", "X", "A"], {"A"}, (3, 5),
                           judged_ids={"X", "A"})
    assert scores["hit@3"] == 1
    assert scores["mrr"] == 1 / 3
    assert scores["p@3"] == 1 / 3
    assert scores["p@5"] == 1 / 5
    assert score_ranking(["A", "A"], {"A"}, (3,))["r@3"] == 1
    assert score_ranking(["A", "A"], {"A"}, (3,))["p@3"] == 1 / 3


def test_empty_miss_and_invalid_k_gold():
    assert score_ranking([], {"A"}, (1, 10))["p@10"] == 0
    assert score_ranking(["X"], {"A"}, (1,))["mrr"] == 0
    with pytest.raises(ValueError):
        score_ranking(["A"], set())
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            score_ranking(["A"], {"A"}, (bad,))


def test_graded_ideal_nonideal_and_duplicate():
    perfect = score_ranking(["A", "B"], {"A", "B"}, (2,),
                            grades={"A": 2, "B": 1})
    reversed_ = score_ranking(["B", "A"], {"A", "B"}, (2,),
                             grades={"A": 2, "B": 1})
    assert perfect["ndcg@2"] == 1
    assert 0 < reversed_["ndcg@2"] < 1
    assert score_ranking(["A", "A"], {"A", "B"}, (2,),
                         grades={"A": 2, "B": 1})["ndcg@2"] < 1
    with pytest.raises(ValueError):
        score_ranking(["A"], {"A"}, grades={"A": 0})


def test_macro_denominator_excludes_unscored_and_ungraded():
    binary = score_ranking(["A"], {"A"}, (1,))
    graded = score_ranking(["X"], {"A"}, (1,), grades={"A": 2})
    avg = macro_average([binary, graded, {}], (1,))
    assert avg["p@1"] == {"mean": 1, "count": 1}
    assert "ndcg@1" not in avg
    assert macro_average([], (1,)) == {}


def test_unjudged_results_do_not_create_false_positives():
    default = score_ranking(["X", "A"], {"A"}, (1, 3), grades={"A": 2})
    assert "p@1" not in default and "ndcg@1" not in default
    assert default["r@3"] == 1 and default["mrr"] == .5
    scores = score_ranking(["X", "A"], {"A"}, (1, 3), judged_ids={"A"})
    assert "p@1" not in scores and "p@3" not in scores
    assert scores["r@3"] == 1
    judged = score_ranking(["X", "A"], {"A"}, (3,), judged_ids={"X", "A"})
    assert judged["p@3"] == 1 / 3
    graded = score_ranking(["X", "A"], {"A"}, (1,),
                           grades={"A": 2}, judged_ids={"X", "A"})
    assert graded["ndcg@1"] == 0
