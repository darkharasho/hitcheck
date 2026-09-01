"""Card image downloader.

Uses `images.small` (~245x342). DINOv2 ingests 224x224, so the hi-res
variant would cost roughly 8GB to gain nothing. Files are sharded by set
id to keep directory sizes sane across ~20k cards.

Resume treats a file as "already downloaded" only if it exists AND is
non-empty. Writes always land via a `.part` temp file plus `os.replace`,
so a process killed mid-write can never leave a half-written file at the
final path in the first place. The size check on resume is a second,
independent line of defense: it also covers a zero-byte or truncated
file that ends up at the final path some other way (an older/buggy
writer, manual tampering, a partial filesystem) — a rerun must not
mistake that for complete and skip it forever.
"""

import os
import time

from .backoff import backoff_delays

RETRYABLE = {0, 429, 500, 502, 503, 504}


def fetch_to_path(url, path, fetch, sleep=time.sleep, max_attempts=4) -> bool:
    """Download one URL to one path, retrying, landing it atomically.

    Returns True once the bytes are at `path`. Writes go to a `.part`
    temp file and arrive via `os.replace`, so a process killed mid-write
    can never leave a half-written file at the final path. An empty body
    counts as a failure: a zero-byte file at the final path would be
    mistaken for a completed download by any later resume check.

    Shared by the catalog sync and the M2 corpus builder. They key files
    differently — card id versus eBay itemId — but the retry schedule and
    the atomic write must not diverge between them.
    """
    delays = backoff_delays(max_attempts - 1)
    for attempt in range(max_attempts):
        status, body = fetch(url)
        if status == 200 and body:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            tmp = f"{path}.part"
            with open(tmp, "wb") as fh:
                fh.write(body)
            os.replace(tmp, path)  # atomic — no half-written images
            return True
        if status not in RETRYABLE:
            return False
        if attempt < len(delays):
            sleep(delays[attempt])
    return False


def image_path(root: str, card_id: str) -> str:
    shard = card_id.rsplit("-", 1)[0] if "-" in card_id else "_"
    return os.path.join(root, shard, f"{card_id}.png")


def download_images(pairs, root, fetch, sleep=time.sleep, max_attempts=4, on_progress=None):
    """Download every (card_id, url) pair that isn't already on disk.

    Returns (downloaded, skipped). Failures are counted in neither —
    they are simply absent, and a rerun retries them.
    """
    downloaded = 0
    skipped = 0
    total = len(pairs)

    for i, (card_id, url) in enumerate(pairs, start=1):
        path = image_path(root, card_id)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            skipped += 1
            if on_progress:
                on_progress(i, total)
            continue

        if fetch_to_path(url, path, fetch, sleep=sleep, max_attempts=max_attempts):
            downloaded += 1

        if on_progress:
            on_progress(i, total)

    return downloaded, skipped
