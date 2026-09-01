"""Retrieval accuracy scoring.

This module produces the number that decides whether an identification
model gets trained at all. Kept pure so the decision is reproducible.
"""

import math
from dataclasses import dataclass, field

SKIP_TRAINING = "SKIP_TRAINING"
TRAIN_REQUIRED = "TRAIN_REQUIRED"
INCONCLUSIVE = "INCONCLUSIVE"

# 95% two-sided normal quantile.
_Z95 = 1.959963984540054


def wilson_interval(hits: int, total: int, z: float = _Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Wilson rather than the textbook normal approximation because the
    quantity being bounded sits near 0.9 with a few hundred samples,
    where the normal approximation's interval runs off the end of [0, 1]
    and is measurably too narrow. Wilson stays inside the unit range by
    construction and is well behaved at 0 and 1 hits.

    An empty sample returns the full unit range: no data is ignorance,
    not a point estimate of zero.
    """
    if total <= 0:
        return (0.0, 1.0)
    p = hits / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half_width = (z / denominator) * math.sqrt(
        p * (1 - p) / total + z * z / (4 * total * total)
    )
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def label_noise_bound(errors: int, sample: int) -> float:
    """Upper 95% bound on the label error rate, from a hand-audited sample.

    Reported alongside accuracy because a mis-resolved label surfaces as a
    retrieval miss that is not one. The bound, not the observed rate, is
    what gets attached to the verdict: zero errors in 50 audited entries
    does not establish zero errors in 500.
    """
    if sample <= 0:
        return 1.0
    return wilson_interval(errors, sample)[1]


@dataclass
class AccuracyReport:
    total: int
    top1: float
    top5: float
    mean_top1_distance: float
    """Mean distance of the top prediction, averaged over queries that
    returned at least one prediction — NOT over `total`. A query with zero
    predictions contributes no distance and is excluded from this average
    (there is nothing to average), even though it still counts as a miss
    in `top1`/`top5`, which are always averaged over `total`. Comparing
    this field across runs with differing numbers of empty-prediction
    queries is therefore misleading; compare it alongside `total` and the
    number of empty-prediction queries (derivable from `failures`), not
    in isolation.
    """
    failures: list[tuple[str, str]] = field(default_factory=list)

    @property
    def interval(self) -> tuple[float, float]:
        """95% Wilson interval on `top1`."""
        return wilson_interval(round(self.top1 * self.total), self.total)

    def verdict(self, threshold: float = 0.90) -> str:
        """Decide on the interval, never on the point estimate alone.

        Returns INCONCLUSIVE when the 95% interval straddles `threshold` —
        the sample cannot resolve which side of the bar it is on. This is
        deliberately a third outcome rather than a rounding rule: a verdict
        flipped by sampling noise is worse than no verdict, because it
        would be acted on. INCONCLUSIVE means collect more corpus.

        At N=500 the decisive bands are top1 >= 0.928 and top1 <= 0.872;
        at N=2000, 0.9135 and 0.8865.
        """
        low, high = self.interval
        if low >= threshold:
            return SKIP_TRAINING
        if high < threshold:
            return TRAIN_REQUIRED
        return INCONCLUSIVE

    def summary(self) -> str:
        low, high = self.interval
        return (
            f"queries={self.total} top1={self.top1:.3f} "
            f"ci95=[{low:.3f}, {high:.3f}] top5={self.top5:.3f} "
            f"mean_top1_distance={self.mean_top1_distance:.4f} verdict={self.verdict()}"
        )


def score(results: list[tuple[str, list[tuple[str, float]]]]) -> AccuracyReport:
    if not results:
        return AccuracyReport(total=0, top1=0.0, top5=0.0, mean_top1_distance=0.0)

    hits1 = 0
    hits5 = 0
    distances: list[float] = []
    failures: list[tuple[str, str]] = []

    for true_id, ranked in results:
        predicted_ids = [card_id for card_id, _ in ranked[:5]]
        if predicted_ids and predicted_ids[0] == true_id:
            hits1 += 1
        else:
            failures.append((true_id, predicted_ids[0] if predicted_ids else ""))
        if true_id in predicted_ids:
            hits5 += 1
        if ranked:
            distances.append(ranked[0][1])

    total = len(results)
    return AccuracyReport(
        total=total,
        top1=hits1 / total,
        top5=hits5 / total,
        mean_top1_distance=sum(distances) / len(distances) if distances else 0.0,
        failures=failures,
    )
