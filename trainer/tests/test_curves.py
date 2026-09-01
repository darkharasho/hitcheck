import json

import pytest

from hitcheck_trainer.augment.curves import (
    Curve,
    CurveBundle,
    interpolate,
    load_bundle,
    nearest,
    save_bundle,
)


def linear_curve():
    return Curve(name="glare", parameter="strength",
                 points=[(0.0, 0.0), (0.5, 0.05), (1.0, 0.10)])


def test_interpolate_recovers_a_calibrated_point_exactly():
    value, saturated = interpolate(linear_curve(), 0.05)
    assert value == pytest.approx(0.5)
    assert saturated is False


def test_interpolate_is_linear_between_calibrated_points():
    value, _ = interpolate(linear_curve(), 0.075)
    assert value == pytest.approx(0.75)


def test_interpolate_clamps_below_the_curve_without_flagging_saturation():
    # Below the first point means "less degraded than anything calibrated",
    # which is a real answer (0.0), not a saturation.
    value, saturated = interpolate(linear_curve(), -0.01)
    assert value == pytest.approx(0.0)
    assert saturated is False


def test_interpolate_flags_saturation_above_the_curve():
    # Above the last point the forward transform has clamped and is flat.
    # The honest report is ">= 1.0", never an extrapolated 1.7.
    value, saturated = interpolate(linear_curve(), 0.4)
    assert value == pytest.approx(1.0)
    assert saturated is True


def test_nearest_snaps_to_the_closest_calibrated_parameter():
    curve = Curve(name="blur", parameter="kernel",
                  points=[(1.0, 0.02), (3.0, 0.20), (5.0, 0.45)])
    assert nearest(curve, 0.19) == pytest.approx(3.0)
    assert nearest(curve, 0.44) == pytest.approx(5.0)
    assert nearest(curve, 0.0) == pytest.approx(1.0)
    assert nearest(curve, 99.0) == pytest.approx(5.0)


def test_a_non_monotone_curve_is_rejected_at_construction():
    # A descriptor that doubles back cannot be inverted: two parameters
    # map to one reading. Catching it here rather than silently returning
    # whichever branch the search happened to land on.
    with pytest.raises(ValueError, match="not monotone"):
        Curve(name="bad", parameter="strength",
              points=[(0.0, 0.0), (0.5, 0.20), (1.0, 0.10)])


def test_a_curve_with_fewer_than_two_points_is_rejected():
    with pytest.raises(ValueError, match="at least two"):
        Curve(name="bad", parameter="strength", points=[(0.0, 0.0)])


def test_bundle_round_trips_through_json(tmp_path):
    bundle = CurveBundle(
        generated_by="test", sample_images=4, seeds=2,
        curves={"glare": linear_curve()},
    )
    path = str(tmp_path / "curves.json")
    save_bundle(bundle, path)
    with open(path) as fh:
        assert json.load(fh)["curves"]["glare"]["parameter"] == "strength"

    loaded = load_bundle(path)
    assert loaded.sample_images == 4
    assert loaded.curves["glare"].points == linear_curve().points


def test_load_bundle_names_the_missing_file_rather_than_failing_open(tmp_path):
    # An estimator running against a bundle that silently defaulted to
    # empty would report every image as undegraded -- a believable wrong
    # answer, which is the worst kind.
    with pytest.raises(FileNotFoundError, match="calibrate"):
        load_bundle(str(tmp_path / "absent.json"))
