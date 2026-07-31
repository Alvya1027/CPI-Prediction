"""Run MoM Siamese pair-structure optimization."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.siamese_mom_pair_optimization import main


if __name__ == "__main__":
    main()
