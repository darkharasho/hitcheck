import pytest

from hitcheck_trainer.catalog.api import CardPage, CatalogApiError
from hitcheck_trainer.catalog.db import card_count, get_sync_state, open_db
from hitcheck_trainer.catalog.sync import sync_catalog


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
