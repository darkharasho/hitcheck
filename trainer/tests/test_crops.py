import numpy as np
import pytest
from PIL import Image, ImageDraw

from hitcheck_trainer.corpus.crops import (
    CARD_SIZE,
    apply_quad,
    load_crops,
    quad_area,
    save_crops,
    validate_quad,
)


def photo_with_marked_card(quad, size=(400, 400)):
    """A white 'desk' with a red 'card' occupying exactly `quad`."""
    image = Image.new("RGB", size, "white")
    ImageDraw.Draw(image).polygon([tuple(p) for p in quad], fill=(255, 0, 0))
    return image


def test_quad_area_of_a_unit_square_is_one():
    assert quad_area([[0, 0], [1, 0], [1, 1], [0, 1]]) == pytest.approx(1.0)


def test_quad_area_is_the_same_whichever_way_round_the_points_go():
    clockwise = [[0, 0], [10, 0], [10, 10], [0, 10]]
    counter = list(reversed(clockwise))
    assert quad_area(clockwise) == pytest.approx(quad_area(counter))


def test_validate_rejects_anything_that_is_not_four_points():
    with pytest.raises(ValueError):
        validate_quad([[0, 0], [1, 0], [1, 1]])
    with pytest.raises(ValueError):
        validate_quad([[0, 0], [1, 0], [1, 1], [0, 1], [2, 2]])


def test_validate_rejects_a_degenerate_quad():
    # Four clicks in nearly the same place is a misclick, not a crop.
    with pytest.raises(ValueError):
        validate_quad([[0, 0], [1, 0], [1, 1], [0, 1]])


def test_validate_accepts_a_real_sized_quad():
    validate_quad([[50, 60], [300, 40], [330, 300], [80, 330]])


def test_apply_quad_unwarps_an_angled_card_to_a_full_frame():
    quad = [[50, 60], [300, 40], [330, 300], [80, 330]]
    cropped = apply_quad(photo_with_marked_card(quad), quad)
    assert cropped.size == CARD_SIZE
    pixels = np.array(cropped)
    # The card now fills the frame: essentially every pixel is the card,
    # not the white desk it was photographed on.
    is_card = (pixels[:, :, 0] > 200) & (pixels[:, :, 1] < 80)
    assert is_card.mean() > 0.98


def test_apply_quad_of_an_axis_aligned_quad_is_a_plain_crop():
    quad = [[100, 100], [300, 100], [300, 380], [100, 380]]
    image = Image.new("RGB", (400, 400), "white")
    ImageDraw.Draw(image).rectangle([100, 100, 300, 380], fill=(0, 0, 255))
    pixels = np.array(apply_quad(image, quad))
    assert (pixels[:, :, 2] > 200).mean() > 0.98


def test_apply_quad_honours_click_order_rather_than_sorting_corners():
    # Feeding the corners rotated by one position must rotate the output.
    # A card photographed at an angle has no meaningful "topmost" corner,
    # so sorting geometrically would silently rotate some crops.
    quad = [[100, 100], [300, 100], [300, 380], [100, 380]]
    image = Image.new("RGB", (400, 400), "white")
    ImageDraw.Draw(image).rectangle([100, 100, 200, 380], fill=(0, 255, 0))
    upright = np.array(apply_quad(image, quad))
    rotated = np.array(apply_quad(image, quad[1:] + quad[:1]))
    assert not np.allclose(upright, rotated)


def test_apply_quad_accepts_a_size_override():
    quad = [[50, 60], [300, 40], [330, 300], [80, 330]]
    assert apply_quad(photo_with_marked_card(quad), quad, size=(64, 89)).size == (64, 89)


def test_apply_quad_output_defaults_to_the_catalog_image_size():
    # Matching catalog images.small keeps the query at the same scale as
    # the gallery it is matched against.
    assert CARD_SIZE == (245, 342)


def test_crops_round_trip_through_disk(tmp_path):
    crops = {"v1|1|0": [[50.0, 60.0], [300.0, 40.0], [330.0, 300.0], [80.0, 330.0]]}
    path = str(tmp_path / "crops.json")
    save_crops(crops, path)
    assert load_crops(path) == crops


def test_loading_a_missing_crops_file_gives_an_empty_mapping(tmp_path):
    assert load_crops(str(tmp_path / "nope.json")) == {}


def test_saving_crops_leaves_no_part_file(tmp_path):
    save_crops({"a": [[0, 0], [1, 0], [1, 1], [0, 1]]}, str(tmp_path / "crops.json"))
    assert [p.name for p in tmp_path.iterdir()] == ["crops.json"]
