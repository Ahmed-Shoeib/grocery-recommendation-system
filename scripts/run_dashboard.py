"""Launch the Streamlit dashboard - a pure FastAPI HTTP client (see
docs/data-mapping.md section 18).

Thin `streamlit run` wrapper for symmetry with the other scripts/*.py
entrypoints - equivalent to running:
    streamlit run src/recommendation/ui/dashboard.py
directly.

This dashboard is a pure HTTP client - it never loads a Two-Tower/ranker
artifact, builds a `RecommendationService`, or touches SQLite/adapters/
TensorFlow itself. `scripts/run_api.py` (FastAPI) MUST
already be running and reachable at `configs/*.yaml: dashboard
.api_base_url` (default `http://localhost:8000`, override via
`RECS_DASHBOARD_API_BASE_URL`) before starting this dashboard - every
recommendation and every piece of user/catalog data it shows comes from
an HTTP call to that running API. This never trains.

Usage:
    python scripts/run_api.py        # start first, in another terminal
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
