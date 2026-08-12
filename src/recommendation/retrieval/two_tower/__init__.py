"""Neural Two-Tower retrieval model (Phase 4).

feature_encoding.py  - UserFeatures/ProductFeatures -> fixed-size tensors
model.py             - User/Item towers + in-batch-softmax TwoTowerModel
splitting.py         - per-user leave-one-out train/val/test split (not temporal)
examples.py          - leakage-safe training example / eval case construction
evaluation.py        - brute-force Recall@K / HitRate@K (Phase 5 adds ANN)
train.py             - end-to-end training orchestrator
serialization.py      - save/load model + encoder + item embeddings for Phase 5
"""
