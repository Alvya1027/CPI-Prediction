"""Run the stabilized shared-projection Siamese residual experiment."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.siamese_shared_projection_residual import main


if __name__ == "__main__":
    main()
