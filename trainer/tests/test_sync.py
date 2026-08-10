import pytest

from hitcheck_trainer.catalog.api import CardPage, CatalogApiError
from hitcheck_trainer.catalog.db import card_count, get_sync_state, open_db
from hitcheck_trainer.catalog.sync import SyncIncompleteError, sync_catalog


class FakeApi:
    """Serves `total` cards in pages of `page_size`."""

    def __init__(self, total, fail_pages=()):
        self.total = total
        self.fail_pages = set(fail_pages)
        self.pages_fetched = []

    def fetch_page(self, page, page_size=250):
        if page in self.fail_pages:
            raise CatalogApiError(f"page {page} exhausted retries")
        self.pages_fetched.append(page)
        start = (page - 1) * page_size
        cards = [
            {"id": f"c-{i}", "name": f"Card {i}"}
            for i in range(start, min(start + page_size, self.total))
        ]
        return CardPage(cards=cards, page=page, total_count=self.total)


def test_syncs_every_card_across_pages(tmp_path):
    conn = open_db(str(tmp_path / "c.sqlite"))
    assert sync_catalog(FakeApi(total=550), conn, page_size=250) == 550
    assert card_count(conn) == 550


def test_stops_when_a_short_page_signals_the_end(tmp_path):
    conn = open_db(str(tmp_path / "c.sqlite"))
    api = FakeApi(total=100)
    sync_catalog(api, conn, page_size=250)
    assert api.pages_fetched == [1]


def test_checkpoints_the_last_completed_page(tmp_path):
    conn = open_db(str(tmp_path / "c.sqlite"))
    sync_catalog(FakeApi(total=550), conn, page_size=250)
    assert get_sync_state(conn, "last_page") == "3"


def test_resumes_from_the_checkpoint(tmp_path):
    conn = open_db(str(tmp_path / "c.sqlite"))
    with pytest.raises(CatalogApiError):
        sync_catalog(FakeApi(total=550, fail_pages={2}), conn, page_size=250)
    assert get_sync_state(conn, "last_page") == "1"

    api = FakeApi(total=550)
    sync_catalog(api, conn, page_size=250)
    assert api.pages_fetched == [2, 3]
    assert card_count(conn) == 550


def test_a_failed_page_does_not_lose_earlier_pages(tmp_path):
    conn = open_db(str(tmp_path / "c.sqlite"))
    with pytest.raises(CatalogApiError):
        sync_catalog(FakeApi(total=1000, fail_pages={3}), conn, page_size=250)
    assert card_count(conn) == 500


def test_reports_progress(tmp_path):
    conn = open_db(str(tmp_path / "c.sqlite"))
    seen = []
    sync_catalog(FakeApi(total=550), conn, page_size=250, on_progress=lambda d, t: seen.append((d, t)))
    assert seen == [(250, 550), (500, 550), (550, 550)]


def test_an_empty_catalog_syncs_zero_cards(tmp_path):
    conn = open_db(str(tmp_path / "c.sqlite"))
    assert sync_catalog(FakeApi(total=0), conn, page_size=250) == 0


class SpuriousEmptyApi:
    """Mimics a `200` with an empty page landing mid-catalog.

    Page 1 legitimately returns a full page out of 550 total. Page 2
    spuriously reports zero cards even though 300 cards are still missing
    -- the kind of malformed-but-200 response the real API is known to
    return.
    """

    def __init__(self):
        self.pages_fetched = []

    def fetch_page(self, page, page_size=250):
        self.pages_fetched.append(page)
        if page == 1:
            cards = [{"id": f"c-{i}", "name": f"Card {i}"} for i in range(250)]
            return CardPage(cards=cards, page=1, total_count=550)
        return CardPage(cards=[], page=page, total_count=550)


def test_a_spurious_empty_page_does_not_report_false_success(tmp_path):
    conn = open_db(str(tmp_path / "c.sqlite"))

    with pytest.raises(SyncIncompleteError):
        sync_catalog(SpuriousEmptyApi(), conn, page_size=250)

    # Page 1's cards are still stored; the sync just refused to claim it
    # finished when it clearly didn't.
    assert card_count(conn) == 250
    assert get_sync_state(conn, "last_page") == "1"


def test_fully_empty_page_is_retried_and_converges(tmp_path):
    """Regression guard: the fully-empty-page path must behave exactly as
    it did before the truncated-page fix -- checkpoint unchanged, and a
    rerun against a healthy API retries that same page and converges."""
    conn = open_db(str(tmp_path / "c.sqlite"))

    with pytest.raises(SyncIncompleteError):
        sync_catalog(SpuriousEmptyApi(), conn, page_size=250)

    assert get_sync_state(conn, "last_page") == "1"
    assert card_count(conn) == 250

    healthy_api = FakeApi(total=550)
    total = sync_catalog(healthy_api, conn, page_size=250)
    assert healthy_api.pages_fetched == [2, 3]
    assert total == 550
    assert card_count(conn) == 550


class TruncatingApi:
    """Page 2 returns only 10 of the 250 cards it should, once.

    Reproduces the reviewer-traced regression: a non-empty but *truncated*
    read that is NOT the genuine end of the catalog (total_count is far
    larger than what's been stored). Unlike a fully empty page, this used
    to slip past the `if result.cards:` guard and seal the checkpoint on a
    page that was never fully read.
    """

    def __init__(self, total=550, page_size=250):
        self.total = total
        self.page_size = page_size
        self.pages_fetched = []

    def fetch_page(self, page, page_size=250):
        self.pages_fetched.append(page)
        start = (page - 1) * page_size
        end = min(start + page_size, self.total)
        if page == 2:
            end = start + 10  # truncated: 10 cards instead of a full page
        cards = [{"id": f"c-{i}", "name": f"Card {i}"} for i in range(start, end)]
        return CardPage(cards=cards, page=page, total_count=self.total)


def test_truncated_page_does_not_seal_the_checkpoint(tmp_path):
    conn = open_db(str(tmp_path / "c.sqlite"))

    with pytest.raises(SyncIncompleteError):
        sync_catalog(TruncatingApi(), conn, page_size=250)

    # Requirement: the checkpoint must NOT advance past the truncated page
    # 2 -- it must still point at the last page that was actually complete
    # (page 1), or a rerun would skip page 2's missing cards forever.
    assert get_sync_state(conn, "last_page") == "1"
    # Page 1 (250) plus the truncated 10 from page 2 are still stored --
    # storing a page's cards is fine; only sealing the checkpoint on it is not.
    assert card_count(conn) == 260


def test_truncated_page_is_retried_until_it_converges(tmp_path):
    conn = open_db(str(tmp_path / "c.sqlite"))

    with pytest.raises(SyncIncompleteError):
        sync_catalog(TruncatingApi(), conn, page_size=250)
    assert card_count(conn) == 260

    # A second run against a now-healthy API must re-fetch page 2 (not skip
    # past it, since the checkpoint never sealed on it) and converge to the
    # full catalog with no exception. This is the convergence property: an
    # unconverged sync must eventually complete once the API cooperates,
    # not loop forever 240 cards short.
    healthy_api = FakeApi(total=550)
    total = sync_catalog(healthy_api, conn, page_size=250)
    assert healthy_api.pages_fetched == [2, 3]
    assert total == 550
    assert card_count(conn) == 550


def test_rerunning_a_complete_sync_makes_no_api_call(tmp_path):
    conn = open_db(str(tmp_path / "c.sqlite"))
    assert sync_catalog(FakeApi(total=550), conn, page_size=250) == 550

    class ExplodingApi:
        """Any call to this API is a bug: the sync should already be done."""

        def fetch_page(self, page, page_size=250):
            raise CatalogApiError(f"page {page} exhausted retries")

    # A rerun of an already-complete catalog must recognize that from
    # persisted state and never touch the (flaky) API again.
    assert sync_catalog(ExplodingApi(), conn, page_size=250) == 550


def test_crash_between_upsert_and_checkpoint_leaves_page_resumable(tmp_path, monkeypatch):
    import hitcheck_trainer.catalog.sync as sync_module

    conn = open_db(str(tmp_path / "c.sqlite"))
    real_set_sync_state = sync_module.set_sync_state

    def flaky_set_sync_state(conn, key, value):
        if key == "last_page":
            raise RuntimeError("simulated crash before checkpoint write")
        return real_set_sync_state(conn, key, value)

    monkeypatch.setattr(sync_module, "set_sync_state", flaky_set_sync_state)

    with pytest.raises(RuntimeError):
        sync_catalog(FakeApi(total=550), conn, page_size=250)

    # The page's cards were stored -- upsert_cards ran before the
    # checkpoint write that crashed.
    assert card_count(conn) == 250
    # But the checkpoint was never advanced, since the crash happened
    # before that write landed.
    assert get_sync_state(conn, "last_page") is None

    # A rerun must not skip page 1: it re-fetches and re-stores it
    # (upsert_cards is idempotent by card id), rather than silently
    # missing 250 cards forever.
    monkeypatch.setattr(sync_module, "set_sync_state", real_set_sync_state)
    api = FakeApi(total=550)
    sync_catalog(api, conn, page_size=250)
    assert api.pages_fetched == [1, 2, 3]
    assert card_count(conn) == 550
