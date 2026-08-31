"""DINOv2 image embeddings.

DINOv2 is used off the shelf, with no fine-tuning. Whether that is good
enough is precisely the question M2 answers — see eval/synthetic.py.
"""

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from .build import normalize


class Embedder:
    def __init__(self, model_name: str = "facebook/dinov2-base", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = AutoImageProcessor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.dim = int(self._model.config.hidden_size)

    @torch.inference_mode()
    def embed(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)

        chunks = []
        for start in range(0, len(images), batch_size):
            batch = [img.convert("RGB") for img in images[start : start + batch_size]]
            inputs = self._processor(images=batch, return_tensors="pt").to(self.device)
            outputs = self._model(**inputs)
            # CLS token — DINOv2's global image descriptor.
            chunks.append(outputs.last_hidden_state[:, 0].float().cpu().numpy())

        return normalize(np.concatenate(chunks, axis=0))
