"""Utility to merge raw CSV files into a single dataset.

This script scans the ``Data/Raw`` directory for ``.csv`` files,
concatenates them and stores the resulting dataset in
``Data/Preprocess/merged_raw.csv``.

The goal is to provide a single entry point for later EDA and
modeling stages when many CSV files are present.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

RAW_DIR = Path("Data/Raw")
OUTPUT_FILE = Path("Data/Preprocess/merged_raw.csv")


def load_csv_files(raw_dir: Path) -> list[pd.DataFrame]:
    """Load all CSV files found in ``raw_dir``.

    Parameters
    ----------
    raw_dir: Path
        Directory containing CSV files.

    Returns
    -------
    list[pandas.DataFrame]
        List with the data loaded from each CSV file. The list is empty
        if no CSVs are found.
    """
    csv_files = sorted(raw_dir.glob("*.csv"))
    return [pd.read_csv(csv_file) for csv_file in csv_files]


def merge_dataframes(dataframes: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate a list of dataframes, resetting the index."""
    if not dataframes:
        raise FileNotFoundError(f"No CSV files were found in {RAW_DIR}")
    return pd.concat(dataframes, ignore_index=True)


def main() -> None:
    dataframes = load_csv_files(RAW_DIR)
    merged_df = merge_dataframes(dataframes)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Merged {len(dataframes)} files into {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
