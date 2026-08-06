"""Run the teacher-specified fixed-reservoir, shared-output-readout experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.teacher_shared_readout_pipeline import (
    DATA_DIR,
    OUTPUT_DIR,
    SERIAL_STATE_DIR,
    run_frozen_test,
    run_validation_search,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--state-dir", type=Path, default=SERIAL_STATE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--frozen-test",
        type=Path,
        help="Evaluate test from an existing validation-frozen JSON config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frozen_test is None:
        config = run_validation_search(
            output_dir=args.output_dir,
            data_dir=args.data_dir,
            state_dir=args.state_dir,
        )
        print(f"Validation configuration frozen: {config}")
        print(args.output_dir / "tables" / "selected_configuration.json")
    else:
        comparison = run_frozen_test(
            args.frozen_test,
            data_dir=args.data_dir,
            state_dir=args.state_dir,
        )
        print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
