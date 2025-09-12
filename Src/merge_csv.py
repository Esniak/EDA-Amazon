"""Merge raw CSV files and produce a cleaned dataset.

The script walks through the ``Data/Raw`` directory, merging every CSV file
found. After concatenation the dataset is cleaned using functions from
``src_data_preparation`` and the resulting dataframe is stored in
``Data/Processed/merged_clean.csv``.

Each processed file name is reported and files missing required columns are
skipped to avoid runtime errors when dealing with heterogeneous data sources.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from src_data_preparation import add_ranking, clean_prices, load_raw_data, split_categories


REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "Data" / "Raw"
MERGED_RAW = REPO_ROOT / "Data" / "Preprocess" / "merged_raw.csv"
CLEAN_OUTPUT = REPO_ROOT / "Data" / "Processed" / "merged_clean.csv"

REQUIRED_COLUMNS = {
    "product_id",
    "product_name",
    "discounted_price",
    "actual_price",
    "discount_percentage",
    "category",
    "rating",
    "rating_count",
}


def iter_csv_files(raw_dir: Path) -> list[pd.DataFrame]:
    """Iterate over CSV files yielding cleanable dataframes.

    Every processed filename is printed. Files lacking mandatory columns are
    skipped.
    """

    dataframes: list[pd.DataFrame] = []
    for csv_file in sorted(raw_dir.glob("*.csv")):
        try:
            df = load_raw_data(csv_file)
        except Exception as exc:  # pragma: no cover - defensive programming
            print(f"Error reading {csv_file.name}: {exc}")
            continue

        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            print(f"Skipping {csv_file.name}: missing columns {missing}")
            continue

        print(f"Processed {csv_file.name}")
        dataframes.append(df)

    return dataframes


def merge_dataframes(dataframes: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate a list of dataframes, resetting the index."""

    if not dataframes:
        raise FileNotFoundError(f"No CSV files were found in {RAW_DIR}")
    return pd.concat(dataframes, ignore_index=True)


def main() -> None:
    dataframes = iter_csv_files(RAW_DIR)
    merged_df = merge_dataframes(dataframes)
    MERGED_RAW.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(MERGED_RAW, index=False)

    cleaned_df = add_ranking(split_categories(clean_prices(merged_df)))
    CLEAN_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(CLEAN_OUTPUT, index=False)

    print(
        f"Merged {len(dataframes)} files with {len(cleaned_df)} rows into {CLEAN_OUTPUT}"
    )


if __name__ == "__main__":
    main()
