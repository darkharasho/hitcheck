"""eBay Item Specifics to a catalog card id.

This is the accuracy-contaminating step of the M2 corpus. A discard costs
one entry; a wrong resolution corrupts the number the whole exercise
exists to produce. So the rule is absolute: accept only unambiguous
agreement on name, set and number, and never guess.

Every discard carries a named reason so build.py can report the corpus's
own yield. A corpus that silently dropped most of its candidates would
skew toward whichever listings happen to have tidy Item Specifics.
"""

import difflib
from dataclasses import dataclass

from .normalize import normalize_name, normalize_number, normalize_set

MISSING_NAME = "MISSING_NAME"
MISSING_SET = "MISSING_SET"
MISSING_NUMBER = "MISSING_NUMBER"
MISSING_LANGUAGE = "MISSING_LANGUAGE"
NOT_ENGLISH = "NOT_ENGLISH"
UNKNOWN_SET = "UNKNOWN_SET"
AMBIGUOUS_SET = "AMBIGUOUS_SET"
NO_SUCH_NUMBER = "NO_SUCH_NUMBER"
NAME_MISMATCH = "NAME_MISMATCH"
AMBIGUOUS_CARD = "AMBIGUOUS_CARD"

DISCARD_REASONS = (
    MISSING_NAME,
    MISSING_SET,
    MISSING_NUMBER,
    MISSING_LANGUAGE,
    NOT_ENGLISH,
    UNKNOWN_SET,
    AMBIGUOUS_SET,
    NO_SUCH_NUMBER,
    NAME_MISMATCH,
    AMBIGUOUS_CARD,
)

# A fuzzy set match must be this close before it is considered at all,
# and must beat the runner-up by this margin before it is accepted.
_SET_CUTOFF = 0.85
_SET_MARGIN = 0.05


@dataclass(frozen=True)
class Resolution:
    card_id: str | None
    reason: str = ""


class CardLookup:
    """Set-name and (set, number) indexes over the catalog."""

    def __init__(
        self,
        set_ids: dict[str, str],
        cards: dict[tuple[str, str], list[tuple[str, str]]],
    ):
        self._set_ids = set_ids
        self._cards = cards

    @classmethod
    def from_conn(cls, conn) -> "CardLookup":
        set_ids: dict[str, str] = {}
        cards: dict[tuple[str, str], list[tuple[str, str]]] = {}
        rows = conn.execute(
            "SELECT id, name, number, set_id, set_name FROM cards ORDER BY id"
        ).fetchall()
        for row in rows:
            if row["set_name"] and row["set_id"]:
                set_ids.setdefault(normalize_set(row["set_name"]), row["set_id"])
            number_key = normalize_number(row["number"] or "")
            if not number_key or not row["set_id"]:
                continue  # unmatchable by number; indexing under "" would collide
            cards.setdefault((row["set_id"], number_key), []).append(
                (row["id"], normalize_name(row["name"] or ""))
            )
        return cls(set_ids, cards)

    def match_set(self, raw: str) -> tuple[str | None, str]:
        """Resolve a seller's set name to a set id, or say why not."""
        key = normalize_set(raw)
        if not key:
            return None, MISSING_SET
        if key in self._set_ids:
            return self._set_ids[key], ""

        candidates = sorted(self._set_ids)
        close = difflib.get_close_matches(key, candidates, n=2, cutoff=_SET_CUTOFF)
        if len(close) == 1:
            return self._set_ids[close[0]], ""
        if len(close) > 1:
            best = difflib.SequenceMatcher(None, key, close[0]).ratio()
            runner_up = difflib.SequenceMatcher(None, key, close[1]).ratio()
            if best - runner_up > _SET_MARGIN:
                return self._set_ids[close[0]], ""
            return None, AMBIGUOUS_SET

        contains = [name for name in candidates if key in name]
        if len(contains) == 1:
            return self._set_ids[contains[0]], ""
        if len(contains) > 1:
            return None, AMBIGUOUS_SET
        return None, UNKNOWN_SET

    def cards_at(self, set_id: str, number_key: str) -> list[tuple[str, str]]:
        return list(self._cards.get((set_id, number_key), []))


def resolve(aspects: dict[str, str], lookup: CardLookup) -> Resolution:
    """Resolve one listing's Item Specifics, or discard it with a reason."""
    language = aspects.get("Language")
    if not language:
        return Resolution(None, MISSING_LANGUAGE)
    if normalize_name(language) != "english":
        return Resolution(None, NOT_ENGLISH)

    name_key = normalize_name(aspects.get("Card Name", ""))
    if not name_key:
        return Resolution(None, MISSING_NAME)
    if not aspects.get("Set"):
        return Resolution(None, MISSING_SET)
    number_key = normalize_number(aspects.get("Card Number", ""))
    if not number_key:
        return Resolution(None, MISSING_NUMBER)

    set_id, reason = lookup.match_set(aspects["Set"])
    if set_id is None:
        return Resolution(None, reason)

    candidates = lookup.cards_at(set_id, number_key)
    if not candidates:
        return Resolution(None, NO_SUCH_NUMBER)

    matching = [card_id for card_id, catalog_name in candidates if catalog_name == name_key]
    if not matching:
        return Resolution(None, NAME_MISMATCH)
    if len(matching) > 1:
        return Resolution(None, AMBIGUOUS_CARD)
    return Resolution(matching[0])
