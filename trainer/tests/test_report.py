from hitcheck_trainer.eval.report import (
    INCONCLUSIVE,
    SKIP_TRAINING,
    TRAIN_REQUIRED,
    label_noise_bound,
    score,
    wilson_interval,
)


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


def hits(n_hits, n_total):
    """`n_total` results of which `n_hits` are top-1 correct."""
    return (
        [("a", [("a", 0.1)])] * n_hits
        + [("b", [("z", 0.1)])] * (n_total - n_hits)
    )


def test_verdict_is_skip_training_when_the_whole_interval_clears_the_threshold():
    # 1900/2000 -> top1 0.950, interval (0.9396, 0.9587). Entirely above 0.90.
    assert score(hits(1900, 2000)).verdict(threshold=0.90) == SKIP_TRAINING


def test_verdict_is_train_required_when_the_whole_interval_is_below_the_threshold():
    # 1760/2000 -> top1 0.880, interval (0.8650, 0.8935). Entirely below 0.90.
    assert score(hits(1760, 2000)).verdict(threshold=0.90) == TRAIN_REQUIRED


def test_verdict_is_inconclusive_when_the_interval_straddles_the_threshold():
    # 1800/2000 -> top1 exactly 0.900, interval (0.8861, 0.9124). Straddles.
    # This is the case the old code answered SKIP_TRAINING with full
    # confidence, off a point estimate it could not actually resolve.
    assert score(hits(1800, 2000)).verdict(threshold=0.90) == INCONCLUSIVE


def test_a_small_sample_is_inconclusive_even_at_a_high_point_estimate():
    # 10/10 is top1 1.000 but the interval is (0.7225, 1.0) — ten queries
    # cannot clear a 0.90 bar. Sample size must beat the threshold, not luck.
    assert score(hits(10, 10)).verdict(threshold=0.90) == INCONCLUSIVE


def test_the_inconclusive_band_at_n500_is_wider_than_the_standard_error():
    # At N=500 the interval is roughly +/-2.8%, not the +/-1.3% standard
    # error, so 0.92 is NOT decisive against a 0.90 threshold.
    assert score(hits(460, 500)).verdict(threshold=0.90) == INCONCLUSIVE
    assert score(hits(475, 500)).verdict(threshold=0.90) == SKIP_TRAINING
    assert score(hits(430, 500)).verdict(threshold=0.90) == TRAIN_REQUIRED


def test_wilson_interval_brackets_the_point_estimate():
    lo, hi = wilson_interval(450, 500)
    assert lo < 0.90 < hi
    assert abs(lo - 0.8706) < 5e-4
    assert abs(hi - 0.9233) < 5e-4


def test_wilson_interval_stays_inside_zero_and_one_at_the_extremes():
    assert wilson_interval(0, 50)[0] == 0.0
    assert wilson_interval(50, 50)[1] == 1.0


def test_wilson_interval_of_an_empty_sample_is_the_full_unit_range():
    # No data means no information, not a point estimate of zero.
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_interval_narrows_as_the_sample_grows():
    small = wilson_interval(90, 100)
    large = wilson_interval(900, 1000)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_report_exposes_its_own_interval():
    report = score(hits(450, 500))
    assert report.interval == wilson_interval(450, 500)


def test_label_noise_bound_is_an_upper_bound_not_the_observed_rate():
    # 2 wrong labels in 50 audited is an observed 4%, but the true rate
    # could plausibly be higher; the bound is what gets reported.
    bound = label_noise_bound(errors=2, sample=50)
    assert bound > 0.04
    assert bound < 0.15


def test_label_noise_bound_of_a_clean_audit_is_still_nonzero():
    # Zero errors in 50 does not prove zero errors in 500.
    assert label_noise_bound(errors=0, sample=50) > 0.0


def test_label_noise_bound_of_an_empty_audit_is_total_ignorance():
    assert label_noise_bound(errors=0, sample=0) == 1.0


def test_summary_reports_the_interval_and_the_verdict():
    text = score(hits(1900, 2000)).summary()
    assert "ci95=[0.940, 0.959]" in text
    assert "verdict=SKIP_TRAINING" in text
