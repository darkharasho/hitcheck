import numpy as np
import pytest
from PIL import Image

from hitcheck_trainer.index.embed import Embedder

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def embedder():
    return Embedder(device="cpu")


def sample(seed=0, size=(120, 168)):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))


def test_embed_returns_one_row_per_image(embedder):
    out = embedder.embed([sample(1), sample(2), sample(3)])
    assert out.shape == (3, embedder.dim)


def test_embeddings_are_unit_norm(embedder):
    out = embedder.embed([sample(1), sample(2)])
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-4)


def test_the_same_image_embeds_identically(embedder):
    a = embedder.embed([sample(5)])
    b = embedder.embed([sample(5)])
    assert np.allclose(a, b, atol=1e-5)


def test_different_images_embed_differently(embedder):
    out = embedder.embed([sample(1), sample(2)])
    assert not np.allclose(out[0], out[1], atol=1e-3)


def test_batching_does_not_change_results(embedder):
    images = [sample(i) for i in range(5)]
    assert np.allclose(embedder.embed(images, batch_size=2), embedder.embed(images, batch_size=5), atol=1e-4)


def test_an_empty_list_returns_an_empty_array(embedder):
    assert embedder.embed([]).shape == (0, embedder.dim)
