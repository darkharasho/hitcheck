"""Memory-bounded embedding of images from disk.

Shared by the synthetic eval and the real-corpus eval. They key their
images differently -- catalog card id versus eBay itemId at an arbitrary
path -- so items are (label, path) pairs and the loader stays one
implementation rather than two.

NEVER materialise a whole gallery. 20,427 images at 240x330 RGB is 4.52GB
of decoded pixels before PIL overhead, degraded copies and torch tensors
-- that allocation contributed to a global OOM on a 30GB machine that
also runs games and browsers. Chunked at 256 the resident set is ~58MB.
Chunking here is a constraint, not an optimisation.
"""

import numpy as np
from PIL import Image


def load_chunk(items, offset=0):
    """Decode one chunk of (label, path) pairs.

    Returns (indices, labels, images) for those that opened, where each
    index is the item's position in the FULL item list -- `offset` is the
    chunk's start. Callers key per-image data off that index, so it has to
    survive a skip.

    An unreadable file is skipped rather than fatal: a truncated download
    must not abort an embed of twenty thousand images, and a catalog rerun
    replaces it.
    """
    indices, labels, images = [], [], []
    for index, (label, path) in enumerate(items, offset):
        try:
            with Image.open(path) as img:
                images.append(img.convert("RGB").copy())
            labels.append(label)
            indices.append(index)
        except OSError:
            continue
    return indices, labels, images


def embed_in_chunks(embedder, items, chunk=256, transform=None):
    """Embed (label, path) pairs a chunk at a time.

    Holds only `chunk` images decoded at once -- see the module docstring
    for why that bound is not negotiable. `transform(image, index)`
    optionally rewrites each image before embedding; `index` is the item's
    position in `items`, not its position within the chunk, because
    callers seed reproducible degradation off it.
    """
    kept_labels, vectors = [], []
    for start in range(0, len(items), chunk):
        indices, labels, images = load_chunk(items[start : start + chunk], offset=start)
        if not labels:
            continue
        if transform is not None:
            images = [transform(img, i) for i, img in zip(indices, images)]
        vectors.append(embedder.embed(images, batch_size=64))
        kept_labels.extend(labels)
        del images  # drop decoded pixels before the next chunk
    if not vectors:
        return kept_labels, np.zeros((0, embedder.dim), dtype=np.float32)
    return kept_labels, np.concatenate(vectors, axis=0)
