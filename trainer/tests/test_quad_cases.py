"""The validator exists twice; this pins the two copies to one truth table.

crops.validate_quad is the authority that gates the corpus. The hosted crop
tool carries a second implementation in JavaScript so a cropper is told
immediately, rather than a fortnight later when sync.pull runs. Two
implementations of the winding contract would drift, and the drift would be
invisible: a mirrored crop is a perfectly valid-looking crop that simply
cannot retrieve its own catalog scan.

Both sides read workers/croptool/quad-cases.json. Changing the rule in one
language now fails a test in the other rather than corrupting ground truth.
The mirror suite is workers/croptool/test/quad.test.js.
"""

import json
import os

import pytest

from hitcheck_trainer.corpus.crops import validate_quad

CASES_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "workers", "croptool", "quad-cases.json"
)


def load_cases():
    with open(CASES_PATH) as fh:
        return json.load(fh)["cases"]


def test_the_shared_case_file_is_reachable_from_here():
    # A moved or renamed worker directory must fail loudly. Silently
    # skipping would leave both suites green while the two validators drift.
    assert os.path.exists(CASES_PATH), f"shared quad cases missing at {CASES_PATH}"
    assert load_cases(), "shared quad case file has no cases"


@pytest.mark.parametrize("case", load_cases(), ids=lambda c: c["why"])
def test_validate_quad_agrees_with_the_shared_truth_table(case):
    if case["valid"]:
        validate_quad(case["quad"])
    else:
        with pytest.raises(ValueError):
            validate_quad(case["quad"])


def test_the_table_covers_both_verdicts():
    # A table that had drifted to all-valid or all-invalid would still pass
    # every case above while testing nothing.
    verdicts = {case["valid"] for case in load_cases()}
    assert verdicts == {True, False}
