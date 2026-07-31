"""Command-line entry point for the MoM strict-closed50 Siamese experiment."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.siamese_mom_closed50_pipeline import main


if __name__ == "__main__":
    main()
