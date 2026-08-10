import pytest
from hitcheck_trainer.catalog.backoff import backoff_delays


def test_returns_one_delay_per_retry():
    assert len(backoff_delays(4)) == 4


def test_delays_double():
    assert backoff_delays(4, base=1.0, cap=100.0) == [1.0, 2.0, 4.0, 8.0]


def test_delays_are_capped():
    assert backoff_delays(5, base=1.0, cap=4.0) == [1.0, 2.0, 4.0, 4.0, 4.0]


def test_zero_attempts_yields_no_delays():
    assert backoff_delays(0) == []


def test_negative_attempts_yields_no_delays():
    assert backoff_delays(-3) == []


def test_rejects_non_positive_base():
    with pytest.raises(ValueError):
        backoff_delays(3, base=0.0)
