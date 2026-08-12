"""Personalized retrieval: Two-Tower model + ANN index.

Submodules:
  two_tower/ - User Tower / Item Tower model, trained in Phase 4.
  index/     - VectorIndex interface with FAISS (default, Windows-native)
               and ScaNN (optional, Linux/Docker) backends, wired in
               Phase 5 and extended in Phase 10.
"""
