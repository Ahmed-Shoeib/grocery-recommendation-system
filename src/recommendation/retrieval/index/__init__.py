"""VectorIndex interface with FAISS (default) and ScaNN (optional) backends.

Both backends do normalized-inner-product search over L2-normalized
128-D Two-Tower embeddings, which is mathematically equivalent to cosine
similarity. Implemented in Phase 5 (FAISS) and Phase 10 (ScaNN, Docker
path only - ScaNN has no Windows wheel).
"""
