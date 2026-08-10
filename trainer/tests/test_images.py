import os

from hitcheck_trainer.catalog.images import download_images, image_path

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64


def test_image_path_shards_by_set_prefix(tmp_path):
    p = image_path(str(tmp_path), "pl3-1")
    assert p.endswith(os.path.join("pl3", "pl3-1.png"))


def test_image_path_handles_ids_without_a_dash(tmp_path):
    assert image_path(str(tmp_path), "weird").endswith(os.path.join("_", "weird.png"))


def test_downloads_and_writes_a_file(tmp_path):
    calls = []

    def fetch(url):
        calls.append(url)
        return 200, PNG

    got, skipped = download_images([("pl3-1", "http://x/1.png")], str(tmp_path), fetch)
    assert (got, skipped) == (1, 0)
    with open(image_path(str(tmp_path), "pl3-1"), "rb") as fh:
        assert fh.read() == PNG
    assert calls == ["http://x/1.png"]


def test_skips_a_file_that_already_exists(tmp_path):
    def fetch(url):
        return 200, PNG

    download_images([("pl3-1", "http://x/1.png")], str(tmp_path), fetch)

    def explode(url):
        raise AssertionError("should not refetch an existing image")

    assert download_images([("pl3-1", "http://x/1.png")], str(tmp_path), explode) == (0, 1)


def test_retries_a_500_then_succeeds(tmp_path):
    responses = [(500, None), (200, PNG)]
    got, _ = download_images(
        [("pl3-1", "http://x/1.png")], str(tmp_path),
        lambda url: responses.pop(0), sleep=lambda _: None,
    )
    assert got == 1


def test_gives_up_after_max_attempts_without_writing_a_file(tmp_path):
    got, skipped = download_images(
        [("pl3-1", "http://x/1.png")], str(tmp_path),
        lambda url: (500, None), sleep=lambda _: None, max_attempts=2,
    )
    assert (got, skipped) == (0, 0)
    assert not os.path.exists(image_path(str(tmp_path), "pl3-1"))


def test_an_empty_body_is_not_written(tmp_path):
    got, _ = download_images(
        [("pl3-1", "http://x/1.png")], str(tmp_path),
        lambda url: (200, b""), sleep=lambda _: None, max_attempts=1,
    )
    assert got == 0
    assert not os.path.exists(image_path(str(tmp_path), "pl3-1"))


def test_one_failure_does_not_stop_the_rest(tmp_path):
    def fetch(url):
        return (500, None) if "bad" in url else (200, PNG)

    got, _ = download_images(
        [("a-1", "http://x/bad.png"), ("a-2", "http://x/good.png")],
        str(tmp_path), fetch, sleep=lambda _: None, max_attempts=1,
    )
    assert got == 1


def test_reports_progress(tmp_path):
    seen = []
    download_images(
        [("a-1", "http://x/1.png"), ("a-2", "http://x/2.png")],
        str(tmp_path), lambda url: (200, PNG),
        on_progress=lambda done, total: seen.append((done, total)),
    )
    assert seen == [(1, 2), (2, 2)]


def test_a_zero_byte_file_on_disk_is_not_treated_as_downloaded(tmp_path):
    """A prior run killed mid-write must not leave a file resume mistakes for complete.

    The atomic os.replace pattern prevents this going forward, but if a
    zero-byte or partial file somehow ends up at the final path (e.g. from
    an older/buggy version, or manual tampering), resume must not trust it
    blindly. We simulate that hazard directly by placing a garbage file at
    the final path and asserting a rerun overwrites it rather than skipping it.
    """
    path = image_path(str(tmp_path), "pl3-1")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(b"")

    calls = []

    def fetch(url):
        calls.append(url)
        return 200, PNG

    got, skipped = download_images([("pl3-1", "http://x/1.png")], str(tmp_path), fetch)

    assert calls == ["http://x/1.png"]
    assert (got, skipped) == (1, 0)
    with open(path, "rb") as fh:
        assert fh.read() == PNG


def test_no_leftover_part_file_after_a_successful_download(tmp_path):
    got, _ = download_images(
        [("pl3-1", "http://x/1.png")], str(tmp_path), lambda url: (200, PNG),
    )
    assert got == 1
    assert not os.path.exists(image_path(str(tmp_path), "pl3-1") + ".part")
