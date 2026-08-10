"""Resumable catalog sync.

The upstream API fails roughly half the time, so a full sync is expected
to be interrupted. Every page is committed and checkpointed before the
next is requested; a rerun picks up where the last one stopped rather
than starting over.
"""

from .db import card_count, get_sync_state, set_sync_state, upsert_cards


class SyncIncompleteError(Exception):
    """Raised when a short/empty page leaves the store short of total_count.

    A page with fewer cards than page_size normally means "this is the
    last page." But this API is flaky enough that a spurious `200` with an
    empty or truncated `data` array can arrive mid-catalog. Treating that
    the same as a genuine last page would silently report success on a
    truncated sync. Whenever a short page still leaves card_count(conn)
    below the total_count the API itself reported, that's unmistakably a
    bad read, not completion -- so this is raised instead of returning.
    """


def sync_catalog(api, conn, page_size: int = 250, on_progress=None) -> int:
    last_page = int(get_sync_state(conn, "last_page") or 0)
    known_total_raw = get_sync_state(conn, "total_count")

    # A prior run already confirmed the store holds every card the API
    # reported (card_count caught up to that run's total_count). Fetching
    # last_page + 1 again would only ever land past the end of the
    # catalog: a page that's certain to be empty and, on an API that fails
    # ~50% of the time, just as likely to burn through retries and raise
    # for a request that could never have returned anything. Skip the
    # network call entirely.
    #
    # This is a live comparison against a number, not a one-way "done"
    # flag: it stops applying the moment card_count(conn) no longer meets
    # total_count (e.g. after a reset). Detecting new cards added to the
    # upstream catalog still requires eventually asking the API again --
    # there is no way to learn about growth without a request -- so that
    # discovery happens via a deliberate full reconciliation (clearing the
    # `last_page`/`total_count` sync_state and resyncing from page 1), not
    # from this incremental-resume path silently re-probing forever.
    if known_total_raw is not None and card_count(conn) >= int(known_total_raw):
        return card_count(conn)

    page = last_page + 1
    total_count: int | None = None

    while True:
        result = api.fetch_page(page, page_size=page_size)
        total_count = result.total_count

        if result.cards:
            upsert_cards(conn, result.cards)
            # The checkpoint may only advance past a page that was either
            # fully read (a full page_size batch) or genuinely final (the
            # store has caught up to total_count). A non-empty page that is
            # short of page_size *and* leaves the store below total_count is
            # a truncated read, not the end of the catalog -- sealing the
            # checkpoint on it would make that page's missing cards
            # permanently unreachable, since paging is by page number and
            # nothing later ever re-covers its range. Storing its cards is
            # still fine and desirable (upsert is idempotent; a re-fetch of
            # the same page just overwrites them), only the checkpoint must
            # wait.
            if len(result.cards) == page_size or card_count(conn) >= total_count:
                set_sync_state(conn, "last_page", str(page))
                set_sync_state(conn, "total_count", str(total_count))
            if on_progress:
                on_progress(card_count(conn), total_count)

        # A short page normally means there is nothing after it -- unless
        # the store is still short of the catalog's own reported total, in
        # which case this was a spurious empty/truncated read, not the end.
        if len(result.cards) < page_size:
            if card_count(conn) < total_count:
                raise SyncIncompleteError(
                    f"page {page} returned a short read "
                    f"({len(result.cards)} of up to {page_size} cards) but "
                    f"only {card_count(conn)} of {total_count} total cards "
                    "are stored; refusing to report success on a possibly "
                    "truncated sync"
                )
            break
        page += 1

    return card_count(conn)
