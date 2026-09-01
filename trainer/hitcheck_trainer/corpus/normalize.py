"""Bridge seller free text to machine-generated catalog fields.

eBay Item Specifics are typed by hand; the catalog is generated. Every
mismatch these functions fail to bridge becomes a retrieval "miss" that
is not one, landing directly underneath the accuracy number the M2
decision rests on. So they are deliberately conservative: they fold away
differences that are certainly cosmetic and nothing else.
"""

import re
import unicodedata

_EMBEDDED_NUMBER = re.compile(r"#?\d+\s*/\s*\S+|#\d+")
_NOT_ALNUM = re.compile(r"[^a-z0-9]+")


def _fold(text: str) -> str:
    """Lowercase, strip accents, map & to 'and', drop everything else.

    The ampersand mapping happens before punctuation is stripped: the
    catalog stores "Black & White" and sellers type "Black and White",
    and stripping first would yield blackwhite versus blackandwhite.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_only = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _NOT_ALNUM.sub("", ascii_only.lower().replace("&", " and "))


def normalize_number(raw: str) -> str:
    """Card number as the catalog stores it: '201/165' -> '201'.

    Leading zeros are stripped only from purely numeric values. Card
    numbers are not integers -- H1, DP01, TG12 and SWSH284 are all real
    catalog values -- so the alphanumeric forms pass through whole.
    """
    if not raw:
        return ""
    head = raw.split("/", 1)[0]
    head = _NOT_ALNUM.sub("", head.lower()).upper()
    if not head:
        return ""
    if head.isdigit():
        return str(int(head))
    return head


def normalize_name(raw: str) -> str:
    """Card name for comparison: 'Charizard ex 199/165' -> 'charizardex'.

    Sellers routinely put the card number in the name field, so embedded
    number tokens come out before folding.
    """
    if not raw:
        return ""
    return _fold(_EMBEDDED_NUMBER.sub(" ", raw))


def normalize_set(raw: str) -> str:
    """Set name for comparison: 'Black & White' -> 'blackandwhite'."""
    if not raw:
        return ""
    return _fold(raw)
