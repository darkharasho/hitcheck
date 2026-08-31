"""Retry delay schedules.

pokemontcg.io was measured at a ~50% 5xx rate on 2026-08-10, so retry
behaviour is a first-class concern rather than an afterthought.
"""


def backoff_delays(attempts: int, base: float = 0.5, cap: float = 30.0) -> list[float]:
    """Exponential backoff delays, one per retry attempt, capped at `cap`."""
    if base <= 0:
        raise ValueError("base must be positive")
    if attempts <= 0:
        return []
    return [min(cap, base * (2**i)) for i in range(attempts)]
