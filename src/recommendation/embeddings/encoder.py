"""Frozen Sentence Transformer wrapper.

Model choice: `all-MiniLM-L6-v2`, 384-D. Chosen over larger options
(e.g. `all-mpnet-base-v2`, 768-D) because for short grocery product/search/
chatbot text, MiniLM's quality is close enough to the larger models while
being roughly 5x smaller (~80MB vs ~420MB) and 4-5x faster on CPU - and V1
serving is CPU-only (see docs/data-mapping.md, ANN backend section, for
the same Windows-CPU-first reasoning applied to retrieval). At this
catalog scale (~50 products) inference cost is negligible either way, but
the model choice must hold up once the catalog is 1000x larger, where
MiniLM's throughput advantage matters. It is never fine-tuned - loaded
once, frozen, `eval()` mode, used purely for inference.
"""

from __future__ import annotations

import numpy as np


class SentenceTransformerEncoder:
    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer  # local import: keep this an optional/heavy dep

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)
        self._model.eval()

    @property
    def embedding_dim(self) -> int:
        # sentence-transformers >=5.1 renamed this; support both.
        if hasattr(self._model, "get_embedding_dimension"):
            return self._model.get_embedding_dimension()
        return self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], normalize: bool = False) -> np.ndarray:
        """Encode a batch of texts. Returns shape (len(texts), embedding_dim).
        Frozen inference only - no gradient tracking, no fine-tuning.
        """
        import torch

        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        with torch.no_grad():
            vectors = self._model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        return np.asarray(vectors, dtype=np.float32)
