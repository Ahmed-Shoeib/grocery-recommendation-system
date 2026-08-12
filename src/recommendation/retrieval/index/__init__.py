"""VectorIndex interface with ScaNN (primary/production) and FAISS
(native-Windows dev fallback) backends.

Both backends do normalized-inner-product search over L2-normalized
128-D Two-Tower embeddings, which is mathematically equivalent to cosine
similarity. ScaNN is the intended production backend but has no Windows
wheel, so it runs only inside the Docker/Linux image
(`configs/docker.yaml` selects it); FAISS runs natively on this Windows
dev machine. Both are implemented in Phase 5 - see
`docs/data-mapping.md` section 10. Full production containerization
(beyond the minimal ScaNN build/test image) lands in Phase 10.

`ScannVectorIndex` is exported lazily (via `__getattr__` below) so that
`import recommendation.retrieval.index` never requires the `scann`
package to be installed - only actually selecting the ScaNN backend does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from recommendation.retrieval.index.base import SearchResult, VectorIndex
from recommendation.retrieval.index.factory import build_vector_index, candidate_pool_size
from recommendation.retrieval.index.faiss_index import FaissVectorIndex

if TYPE_CHECKING:
    from recommendation.retrieval.index.scann_index import ScannVectorIndex

__all__ = [
    "VectorIndex",
    "SearchResult",
    "FaissVectorIndex",
    "ScannVectorIndex",
    "build_vector_index",
    "candidate_pool_size",
]


def __getattr__(name: str):
    if name == "ScannVectorIndex":
        from recommendation.retrieval.index.scann_index import ScannVectorIndex

        return ScannVectorIndex
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
