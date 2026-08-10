"""Resumable catalog sync.

The upstream API fails roughly half the time, so a full sync is expected
to be interrupted. Every page is committed and checkpointed before the
next is requested; a rerun picks up where the last one stopped rather
than starting over.
"""

from .db import card_count, get_sync_state, set_sync_state, upsert_cards


def sync_catalog(api, conn, page_size: int = 250, on_progress=None) -> int:
    last_page = int(get_sync_state(conn, "last_page") or 0)
    page = last_page + 1

    while True:
        result = api.fetch_page(page, page_size=page_size)

        if result.cards:
            upsert_cards(conn, result.cards)
            set_sync_state(conn, "last_page", str(page))
            if on_progress:
                on_progress(card_count(conn), result.total_count)

        # A short page means there is nothing after it.
        if len(result.cards) < page_size:
            break
        page += 1

    return card_count(conn)
