"""Launch the Phase 9 Streamlit dashboard.

Thin `streamlit run` wrapper for symmetry with the other scripts/*.py
entrypoints - equivalent to running:
    streamlit run src/recommendation/ui/dashboard.py
directly.

Requires the Phase 4 Two-Tower and Phase 6 ranker artifacts to already
exist under `models/` (`scripts/train_two_tower.py` then
`scripts/train_ranker.py`) - the dashboard loads them once, in-process,
via the same `RecommendationService` the API uses; it never trains.

Usage:
    python scripts/run_dashboard.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    dashboard_path = Path(__file__).resolve().parents[1] / "src" / "recommendation" / "ui" / "dashboard.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)], check=True)


if __name__ == "__main__":
    main()
