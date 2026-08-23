"""Neural Two-Tower retrieval model.

feature_encoding.py  - UserFeatures/ProductFeatures -> fixed-size tensors
model.py             - User/Item towers + in-batch-softmax TwoTowerModel
splitting.py         - per-user leave-one-out train/val/test split (not temporal)
examples.py          - leakage-safe training example / eval case construction
evaluation.py        - brute-force Recall@K / HitRate@K (ANN retrieval lives in retrieval.index)
train.py             - end-to-end training orchestrator
serialization.py      - save/load model + encoder + item embeddings for retrieval.index
"""
