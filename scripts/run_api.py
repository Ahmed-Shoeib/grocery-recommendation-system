"""Run the Phase 8 recommendation API with uvicorn.

Loads the ALREADY-TRAINED Two-Tower (Phase 4) and ranker (Phase 6)
artifacts once at startup (does not retrain either) - run
scripts/train_two_tower.py and scripts/train_ranker.py first if those
don't exist yet.

Usage:
    python scripts/run_api.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import uvicorn

from recommendation.api.app import create_app
from recommendation.utils.logging import setup_logging

setup_logging()
app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
