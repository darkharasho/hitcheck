"""Retrieval accuracy scoring.

This module produces the number that decides whether an identification
model gets trained at all. Kept pure so the decision is reproducible.
"""

from dataclasses import dataclass, field

SKIP_TRAINING = "SKIP_TRAINING"
TRAIN_REQUIRED = "TRAIN_REQUIRED"


@dataclass
class AccuracyReport:
    total: int
    top1: float
    top5: float
    mean_top1_distance: float
    failures: list[tuple[str, str]] = field(default_factory=list)

    def verdict(self, threshold: float = 0.90) -> str:
        """Above threshold, zero-shot retrieval is good enough to ship."""
        return SKIP_TRAINING if self.top1 >= threshold else TRAIN_REQUIRED

    def summary(self) -> str:
        return (
            f"queries={self.total} top1={self.top1:.3f} top5={self.top5:.3f} "
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
