"""pokemontcg.io v2 client.

Only the paged /v2/cards list endpoint is used. /v2/sets and
/v2/cards/{id} were both returning 500 on 2026-08-10; set metadata is
taken from the `set` object embedded in each card instead.
"""

import time

from .backoff import backoff_delays

BASE_URL = "https://api.pokemontcg.io/v2/cards"

# Statuses worth another attempt. Everything else is a hard failure.
RETRYABLE = {0, 429, 500, 502, 503, 504}


class CatalogApiError(Exception):
    pass


class CardPage:
    def __init__(self, cards: list[dict], page: int, total_count: int):
        self.cards = cards
        self.page = page
        self.total_count = total_count


class CatalogApi:
    def __init__(self, transport, api_key: str | None = None, max_attempts: int = 6, sleep=time.sleep):
        self._transport = transport
        self._api_key = api_key
        self._max_attempts = max_attempts
        self._sleep = sleep

    def fetch_page(self, page: int, page_size: int = 250) -> CardPage:
        url = f"{BASE_URL}?page={page}&pageSize={page_size}"
        headers: dict[str, str] = {}
        if self._api_key:
            headers["X-Api-Key"] = self._api_key

        delays = backoff_delays(self._max_attempts - 1)
        last_status: int | None = None

        for attempt in range(self._max_attempts):
            status, body = self._transport(url, headers)
            last_status = status
            if status == 200 and body is not None:
                return CardPage(
                    cards=body.get("data", []),
                    page=body.get("page", page),
                    total_count=body.get("totalCount", 0),
                )
            if status not in RETRYABLE:
                break
            if attempt < len(delays):
                self._sleep(delays[attempt])

        raise CatalogApiError(f"page {page} failed after {self._max_attempts} attempts (last status {last_status})")
