from hitcheck_trainer.eval.report import score


def preds(*pairs):
    return list(pairs)


def test_perfect_top1():
    report = score([("a", preds(("a", 0.01), ("b", 0.5)))])
    assert report.top1 == 1.0
    assert report.top5 == 1.0


def test_counts_a_top5_hit_that_is_not_top1():
    report = score([("a", preds(("b", 0.1), ("c", 0.2), ("a", 0.3)))])
    assert report.top1 == 0.0
    assert report.top5 == 1.0


def test_a_complete_miss_scores_zero():
    report = score([("a", preds(("b", 0.1), ("c", 0.2)))])
    assert report.top1 == 0.0
    assert report.top5 == 0.0


def test_averages_across_queries():
    report = score([
        ("a", preds(("a", 0.1))),
        ("b", preds(("x", 0.1))),
        ("c", preds(("c", 0.1))),
        ("d", preds(("y", 0.1))),
    ])
    assert report.top1 == 0.5
    assert report.total == 4


def test_a_sixth_place_hit_does_not_count_as_top5():
    ranked = preds(*[(f"x{i}", 0.1 * i) for i in range(5)], ("a", 0.9))
    assert score([("a", ranked)]).top5 == 0.0


def test_records_failures_as_true_predicted_pairs():
    report = score([("a", preds(("b", 0.1))), ("c", preds(("c", 0.1)))])
    assert report.failures == [("a", "b")]


def test_mean_top1_distance_is_averaged_over_all_queries():
    report = score([("a", preds(("a", 0.2))), ("b", preds(("b", 0.4)))])
    assert abs(report.mean_top1_distance - 0.3) < 1e-9


def test_empty_results_are_zeroed_not_a_crash():
    report = score([])
    assert report.total == 0 and report.top1 == 0.0


def test_a_query_with_no_predictions_counts_as_a_miss():
    report = score([("a", [])])
    assert report.top1 == 0.0
    assert report.failures == [("a", "")]


def test_verdict_is_skip_training_above_the_threshold():
    report = score([("a", preds(("a", 0.1)))] * 10)
    assert report.verdict(threshold=0.90) == "SKIP_TRAINING"


def test_verdict_is_train_required_below_the_threshold():
    results = [("a", preds(("a", 0.1)))] * 8 + [("b", preds(("z", 0.1)))] * 2
    assert score(results).verdict(threshold=0.90) == "TRAIN_REQUIRED"


def test_verdict_boundary_exact_threshold_is_skip_training():
    # 9 of 10 queries hit top-1 -> top1 == 0.9 exactly (9/10 and the
    # literal 0.90 round to the identical double), landing precisely on
    # the default threshold. The boundary must be inclusive.
    results = [("a", preds(("a", 0.1)))] * 9 + [("b", preds(("z", 0.1)))] * 1
    report = score(results)
    assert report.top1 == 0.9
    assert report.verdict(threshold=0.90) == "SKIP_TRAINING"


def test_verdict_just_below_threshold_is_train_required():
    # 89 of 100 queries hit top-1 -> top1 == 0.89, comfortably below the
    # 0.90 threshold by a margin far larger than float rounding error.
    results = [("a", preds(("a", 0.1)))] * 89 + [("b", preds(("z", 0.1)))] * 11
    report = score(results)
    assert report.top1 == 0.89
    assert report.verdict(threshold=0.90) == "TRAIN_REQUIRED"


def test_mixed_batch_top1_and_top5_divide_by_total_not_by_answered_queries():
    # Of 3 queries, only 2 return any prediction at all. If top1/top5 were
    # (incorrectly) divided by the count of answered queries instead of
    # `total`, this would silently produce 1/2 == 0.5 instead of 1/3.
    results = [
        ("a", preds(("a", 0.1))),  # top-1 and top-5 hit
        ("b", []),  # no predictions at all -> miss
        ("c", preds(("z", 0.2))),  # predicted wrong card -> miss
    ]
    report = score(results)
    assert report.total == 3
    assert abs(report.top1 - (1 / 3)) < 1e-9
    assert abs(report.top5 - (1 / 3)) < 1e-9


def test_mixed_batch_mean_top1_distance_averages_only_over_answered_queries():
    # Same mixed batch as above. The no-prediction query for "b"
    # contributes no distance, so the mean is taken over the 2 queries
    # that did return a prediction (0.1 and 0.2), not over all 3 queries.
    results = [
        ("a", preds(("a", 0.1))),
        ("b", []),
        ("c", preds(("z", 0.2))),
    ]
    report = score(results)
    assert abs(report.mean_top1_distance - 0.15) < 1e-9
