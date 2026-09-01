"""The parts of corpus sync that decide what enters the ground truth.

The HTTP is not tested here -- it is thin, and a wrong URL fails loudly on
first use. What is tested is the merge, because a bug in it is silent: a
mirrored or slab-shaped quad that slips through does not raise anything, it
just moves the M2 accuracy number.
"""

import pytest

from hitcheck_trainer.corpus.sync import (
    calibration_errors,
    corner_disagreement,
    merge_remote_crops,
)

REFERENCE = [[10.0, 20.0], [210.0, 20.0], [210.0, 320.0], [10.0, 320.0]]
MIRRORED = [[10.0, 320.0], [210.0, 320.0], [210.0, 20.0], [10.0, 20.0]]


def crop(item_id, cropper="a@b.c", quad=None, at=1):
    return {"item_id": item_id, "cropper": cropper, "quad": quad or REFERENCE, "at": at}


def payload(crops, calibration=(), skips=()):
    return {"crops": list(crops), "calibration": list(calibration), "skips": list(skips)}


def test_an_identical_quad_disagrees_with_the_reference_by_nothing():
    assert corner_disagreement(REFERENCE, REFERENCE) == pytest.approx(0.0)


def test_disagreement_is_scale_free():
    # Expressed against the card's own diagonal so the same threshold works
    # on a 600px phone photo and a 2000px one.
    shifted = [[x + 10, y] for x, y in REFERENCE]
    doubled = [[x * 2, y * 2] for x, y in REFERENCE]
    doubled_shifted = [[x + 20, y] for x, y in doubled]
    assert corner_disagreement(REFERENCE, shifted) == pytest.approx(
        corner_disagreement(doubled, doubled_shifted)
    )


def test_a_quad_started_from_the_wrong_corner_is_a_large_disagreement():
    # The failure no geometric check can catch: same four corners, rotated
    # one position, crops sideways. It must show up loudly in calibration
    # because nothing else will ever flag it.
    rotated = REFERENCE[1:] + REFERENCE[:1]
    assert corner_disagreement(REFERENCE, rotated) > 0.5


def test_calibration_errors_are_grouped_by_cropper_and_ignore_other_cards():
    errors = calibration_errors(
        [crop("cal-1", "a@b.c"), crop("cal-1", "d@e.f"), crop("corpus-1", "a@b.c")],
        {"cal-1"},
        {"cal-1": REFERENCE},
    )
    assert set(errors) == {"a@b.c", "d@e.f"}
    assert len(errors["a@b.c"]) == 1


def test_a_calibration_card_with_no_local_reference_is_not_scored():
    # Scoring against a missing reference would either crash or invent a
    # number; the honest answer is that this card cannot be assessed.
    assert calibration_errors([crop("cal-1")], {"cal-1"}, {}) == {}


def test_a_valid_remote_quad_is_merged():
    merged, rejected, duplicates = merge_remote_crops(payload([crop("i-1")]), {})
    assert merged == {"i-1": REFERENCE}
    assert rejected == [] and duplicates == 0


def test_an_invalid_remote_quad_is_rejected_with_its_cropper_named():
    # Whose pass produced it is the actionable part: one bad cropper's work
    # can then be pulled without redoing everyone else's.
    merged, rejected, _ = merge_remote_crops(
        payload([crop("i-1", "bad@x.y", quad=MIRRORED)]), {}
    )
    assert merged == {}
    assert len(rejected) == 1
    item_id, cropper, reason = rejected[0]
    assert (item_id, cropper) == ("i-1", "bad@x.y")
    assert "counter-clockwise" in reason


def test_calibration_cards_are_never_merged_into_the_corpus():
    # They are measurement, not data: several people mark the same card on
    # purpose, so merging them would put one arbitrary person's quad in.
    merged, _, _ = merge_remote_crops(
        payload([crop("cal-1", "a@b.c"), crop("cal-1", "d@e.f")], calibration=["cal-1"]), {}
    )
    assert merged == {}


def test_a_local_crop_is_not_overwritten_by_a_remote_one():
    # Local quads are the references everyone is measured against. A remote
    # re-mark must not quietly redefine the yardstick.
    local = {"i-1": REFERENCE}
    other = [[11.0, 21.0], [211.0, 21.0], [211.0, 321.0], [11.0, 321.0]]
    merged, _, _ = merge_remote_crops(payload([crop("i-1", quad=other)]), local)
    assert merged["i-1"] == REFERENCE


def test_two_people_on_one_corpus_card_keeps_the_earliest_and_is_counted():
    # Happens when a lease expires while someone is still marking. The
    # second quad is not a correction, so the first one stands -- but the
    # count is reported so repeated collisions surface a lease that is
    # too short.
    other = [[11.0, 21.0], [211.0, 21.0], [211.0, 321.0], [11.0, 321.0]]
    merged, _, duplicates = merge_remote_crops(
        payload([crop("i-1", "late@x.y", quad=other, at=99), crop("i-1", "early@x.y", at=1)]),
        {},
    )
    assert merged["i-1"] == REFERENCE
    assert duplicates == 1
