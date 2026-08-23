"""Run the recommendation API with uvicorn.

Loads the already-trained Two-Tower and ranker artifacts once at startup
(does not retrain either) - run scripts/train_two_tower.py and
scripts/train_ranker.py first if those don't exist yet.

Usage:
    python scripts/run_api.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import uvicorn

from recommendation.api.app import create_app
from recommendation.utils.config import get_config
from recommendation.utils.logging import setup_logging

_config = get_config()
setup_logging(_config.log_level)
app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host=_config.api.host, port=_config.api.port)
