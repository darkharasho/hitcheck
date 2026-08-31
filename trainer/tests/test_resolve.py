from hitcheck_trainer.catalog.db import open_db, upsert_cards
from hitcheck_trainer.corpus.resolve import (
    AMBIGUOUS_CARD,
    AMBIGUOUS_SET,
    DISCARD_REASONS,
    MISSING_LANGUAGE,
    MISSING_NAME,
    MISSING_NUMBER,
    MISSING_SET,
    NAME_MISMATCH,
    NO_SUCH_NUMBER,
    NOT_ENGLISH,
    UNKNOWN_SET,
    CardLookup,
    resolve,
)


def lookup():
    """A small stand-in catalog covering the cases that matter."""
    return CardLookup(
        set_ids={
            "151": "sv3pt5",
            "blackandwhite": "bw1",
            "astralradiance": "swsh10",
            "astralradiancetrainergallery": "swsh10tg",
        },
        cards={
            ("sv3pt5", "6"): [("sv3pt5-6", "charizardex")],
            ("sv3pt5", "199"): [("sv3pt5-199", "charizardex")],
            ("bw1", "1"): [("bw1-1", "snivy")],
            ("swsh10", "TG12"): [("swsh10-TG12", "sylveonvstar")],
        },
    )


def aspects(**overrides):
    base = {
        "Card Name": "Charizard ex",
        "Set": "151",
        "Card Number": "199/165",
        "Language": "English",
    }
    base.update(overrides)
    return {k: v for k, v in base.items() if v is not None}


def test_resolves_a_complete_unambiguous_listing():
    result = resolve(aspects(), lookup())
    assert result.card_id == "sv3pt5-199"
    assert result.reason == ""


def test_resolves_an_alphanumeric_card_number():
    result = resolve(
        aspects(**{"Card Name": "Sylveon VSTAR", "Set": "Astral Radiance",
                   "Card Number": "TG12/TG30"}),
        lookup(),
    )
    assert result.card_id == "swsh10-TG12"


def test_resolves_through_the_ampersand_spelling_difference():
    result = resolve(
        aspects(**{"Card Name": "Snivy", "Set": "Black and White",
                   "Card Number": "1/114"}),
        lookup(),
    )
    assert result.card_id == "bw1-1"


def test_a_missing_aspect_discards_with_a_named_reason():
    assert resolve(aspects(**{"Card Name": None}), lookup()).reason == MISSING_NAME
    assert resolve(aspects(Set=None), lookup()).reason == MISSING_SET
    assert resolve(aspects(**{"Card Number": None}), lookup()).reason == MISSING_NUMBER
    assert resolve(aspects(Language=None), lookup()).reason == MISSING_LANGUAGE


def test_a_missing_language_is_a_discard_not_an_assumed_english():
    # Japanese prints share artwork with English ones but share no
    # numbering, so an unmarked Japanese listing would resolve to a
    # plausible-looking wrong id.
    assert resolve(aspects(Language=None), lookup()).card_id is None


def test_a_non_english_listing_is_discarded():
    result = resolve(aspects(Language="Japanese"), lookup())
    assert result.card_id is None
    assert result.reason == NOT_ENGLISH


def test_a_set_with_no_catalog_match_is_discarded():
    result = resolve(aspects(Set="Totally Invented Set"), lookup())
    assert result.reason == UNKNOWN_SET


def test_a_set_name_contained_in_two_sets_is_ambiguous_not_a_coin_flip():
    # "Astral" is a substring of both "Astral Radiance" and "Astral
    # Radiance Trainer Gallery" -- different sets with different cards --
    # and is not close enough to either for the fuzzy match to fire.
    result = resolve(aspects(Set="Astral"), lookup())
    assert result.card_id is None
    assert result.reason == AMBIGUOUS_SET


def test_a_clear_fuzzy_winner_is_accepted_rather_than_discarded():
    # "Astral Radiance Trainer" scores 0.857 against the Trainer Gallery
    # and 0.80 against plain Astral Radiance, so only one candidate clears
    # the 0.85 cutoff. Being strict does not mean rejecting everything --
    # it means never picking between two candidates that both qualify.
    result = resolve(
        aspects(**{"Card Name": "Sylveon VSTAR", "Set": "Astral Radiance",
                   "Card Number": "TG12/TG30"}),
        lookup(),
    )
    assert result.card_id == "swsh10-TG12"


def test_a_number_absent_from_the_matched_set_is_discarded():
    result = resolve(aspects(**{"Card Number": "9999/165"}), lookup())
    assert result.reason == NO_SUCH_NUMBER


def test_a_name_that_disagrees_with_the_catalog_is_discarded():
    # Set and number both point at sv3pt5-199, but the seller named a
    # different card. This is exactly where guessing would put a wrong
    # label under the accuracy number.
    result = resolve(aspects(**{"Card Name": "Blastoise ex"}), lookup())
    assert result.card_id is None
    assert result.reason == NAME_MISMATCH


def test_two_catalog_cards_at_the_same_set_and_number_are_ambiguous():
    lk = CardLookup(
        set_ids={"151": "sv3pt5"},
        cards={("sv3pt5", "199"): [("sv3pt5-199", "charizardex"),
                                   ("sv3pt5-199a", "charizardex")]},
    )
    assert resolve(aspects(), lk).reason == AMBIGUOUS_CARD


def test_every_reason_a_resolution_can_return_is_listed_in_discard_reasons():
    # build.py tallies discards by iterating DISCARD_REASONS; a reason
    # missing from it would be silently dropped from the yield report.
    for reason in (MISSING_NAME, MISSING_SET, MISSING_NUMBER, MISSING_LANGUAGE,
                   NOT_ENGLISH, UNKNOWN_SET, AMBIGUOUS_SET, NO_SUCH_NUMBER,
                   NAME_MISMATCH, AMBIGUOUS_CARD):
        assert reason in DISCARD_REASONS


def test_lookup_from_conn_indexes_a_real_catalog_database(tmp_path):
    conn = open_db(str(tmp_path / "catalog.sqlite"))
    upsert_cards(conn, [
        {"id": "sv3pt5-199", "name": "Charizard ex", "number": "199",
         "set": {"id": "sv3pt5", "name": "151"}, "images": {"small": "http://x/1.png"}},
        {"id": "bw1-1", "name": "Snivy", "number": "1",
         "set": {"id": "bw1", "name": "Black & White"}, "images": {"small": "http://x/2.png"}},
    ])
    lk = CardLookup.from_conn(conn)
    assert lk.match_set("151") == ("sv3pt5", "")
    assert lk.match_set("Black and White") == ("bw1", "")
    assert lk.cards_at("sv3pt5", "199") == [("sv3pt5-199", "charizardex")]


def test_lookup_from_conn_skips_rows_with_no_number(tmp_path):
    # A card with a NULL number can never be matched by number, and
    # indexing it under "" would let a listing with an unparseable number
    # collide with it.
    conn = open_db(str(tmp_path / "catalog.sqlite"))
    upsert_cards(conn, [
        {"id": "bp-1", "name": "Best Of Promo", "number": None,
         "set": {"id": "bp", "name": "Best of Game"}, "images": {"small": "http://x/3.png"}},
    ])
    lk = CardLookup.from_conn(conn)
    assert lk.cards_at("bp", "") == []
