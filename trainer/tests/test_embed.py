import numpy as np
import pytest
import torch
from PIL import Image

from hitcheck_trainer.index.build import normalize
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


def test_the_descriptor_is_the_cls_token_not_mean_pooled_patches(embedder):
    """Pins WHICH descriptor embed() returns.

    Shape, unit-norm, reproducibility, batching-invariance and "different
    images differ" all hold just as well for mean-pooled patch tokens as for
    the CLS token — none of the other tests can catch the descriptor being
    silently swapped. This test asserts embed()'s output matches the model's
    own `pooler_output` for the same input (both L2-normalised), and that it
    is NOT the mean-pooled patch tokens, which is the specific
    plausible-but-wrong alternative.
    """
    img = sample(7)
    out = embedder.embed([img])[0]

    with torch.inference_mode():
        inputs = embedder._processor(images=[img.convert("RGB")], return_tensors="pt").to(
            embedder.device
        )
        outputs = embedder._model(**inputs)

    pooled = normalize(outputs.pooler_output.float().cpu().numpy())[0]
    mean_patches = normalize(outputs.last_hidden_state[:, 1:].mean(dim=1).float().cpu().numpy())[0]

    assert np.allclose(out, pooled, atol=1e-5)
    assert not np.allclose(out, mean_patches, atol=1e-3)
