from hitcheck_trainer.corpus.normalize import (
    normalize_name,
    normalize_number,
    normalize_set,
)


def test_number_drops_the_set_total():
    assert normalize_number("201/165") == "201"


def test_number_keeps_alphanumeric_prefixes_whole():
    # H1, DP01, TG12 and SWSH284 all exist in the catalog; treating card
    # numbers as integers would lose every one of them.
    assert normalize_number("TG12/TG30") == "TG12"
    assert normalize_number("SWSH284") == "SWSH284"
    assert normalize_number("H1") == "H1"


def test_number_strips_leading_zeros_only_when_it_is_purely_numeric():
    assert normalize_number("006/165") == "6"
    assert normalize_number("DP01") == "DP01"


def test_number_strips_whitespace_and_a_leading_hash():
    assert normalize_number("  #199 ") == "199"


def test_number_uppercases_so_case_never_decides_a_match():
    assert normalize_number("tg12") == "TG12"


def test_number_of_junk_is_empty_not_a_guess():
    assert normalize_number("") == ""
    assert normalize_number("   ") == ""
    assert normalize_number("/165") == ""


def test_name_lowercases_and_drops_punctuation_and_spaces():
    assert normalize_name("Charizard ex") == "charizardex"
    assert normalize_name("Professor's Research") == "professorsresearch"


def test_name_drops_a_number_the_seller_put_in_the_name_field():
    assert normalize_name("Charizard ex 199/165") == "charizardex"
    assert normalize_name("Pikachu #58") == "pikachu"


def test_name_maps_ampersand_to_and():
    assert normalize_name("Team Magma & Team Aqua") == "teammagmaandteamaqua"


def test_name_folds_accents_so_pokemon_matches_pokemon():
    assert normalize_name("Pokémon Center Lady") == "pokemoncenterlady"


def test_name_of_junk_is_empty():
    assert normalize_name("") == ""
    assert normalize_name("   -- ") == ""


def test_set_maps_ampersand_to_and_before_stripping_punctuation():
    # The catalog stores "Black & White"; a seller types "Black and White".
    # Stripping punctuation first would give blackwhite vs blackandwhite,
    # which never meet.
    assert normalize_set("Black & White") == normalize_set("Black and White")
    assert normalize_set("Black & White") == "blackandwhite"


def test_set_lowercases_and_drops_spaces_and_punctuation():
    assert normalize_set("Astral Radiance Trainer Gallery") == "astralradiancetrainergallery"
    assert normalize_set("Base Set 2") == "baseset2"


def test_set_keeps_bare_numeric_names():
    # The catalog set sv3pt5 is literally named "151".
    assert normalize_set("151") == "151"


def test_set_of_junk_is_empty():
    assert normalize_set("") == ""
    assert normalize_set(" & ") == "and"
