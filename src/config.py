from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = ROOT_DIR / "data_raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data_processed"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
DOCS_DIR = ROOT_DIR / "docs"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

RANDOM_SEED = 42
DEFAULT_WINDOW_SIZE = 12
DEFAULT_HORIZON = 1