"""Utilities for loading and cleaning raw Amazon datasets.

This module exposes small functions that operate on :class:`~pathlib.Path`
objects and :class:`~pandas.DataFrame` instances so they can be reused across
different scripts. None of the functions perform any I/O besides the
``load_raw_data`` helper; every function returns a new dataframe with the
transformations applied.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load a CSV file containing the raw Amazon data.

    Parameters
    ----------
    path:
        Location of the CSV file.

    Returns
    -------
    pandas.DataFrame
        Dataframe with the contents of ``path``.
    """

    return pd.read_csv(path)


def clean_prices(df: pd.DataFrame, exchange_rate: float = 0.012) -> pd.DataFrame:
    """Clean monetary columns and related fields.

    The function removes currency symbols, converts values to floats, applies a
    currency conversion and computes the discount amount and percentage in
    decimal form. Rating information is also normalised.

    Parameters
    ----------
    df:
        Data to clean. It is not modified in place.
    exchange_rate:
        Exchange rate applied to the price columns. Defaults to ``0.012``.

    Returns
    -------
    pandas.DataFrame
        New dataframe with the transformations applied.
    """

    df = df.copy()

    # Clean price columns
    df["discounted_price"] = (
        df["discounted_price"].str.replace("₹", "").str.replace(",", "").astype("float64")
    )
    df["actual_price"] = (
        df["actual_price"].str.replace("₹", "").str.replace(",", "").astype("float64")
    )

    df["discounted_price"] *= exchange_rate
    df["actual_price"] *= exchange_rate

    df["discounted_price"] = df["discounted_price"].round(2)
    df["actual_price"] = df["actual_price"].round(2)

    df["discount_amount"] = df["actual_price"] - df["discounted_price"]

    df["discount_percentage"] = (
        df["discount_percentage"].str.replace("%", "").astype("float64") / 100
    )

    # Normalise rating information
    df["rating"] = df["rating"].str.replace("|", "3.9").astype("float64")
    df["rating_count"] = df["rating_count"].str.replace(",", "").astype("float64")

    return df


def split_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Split the ``category`` column into ``category`` and ``subcategory``.

    Both resulting columns are formatted to improve readability.
    """

    df = df.copy()

    splitcategory = df["category"].str.split("|", expand=True).rename(
        columns={0: "category", 1: "subcategory"}
    )

    # Category replacements
    replacements_cat = {
        "&": " & ",
        "MusicalInstruments": "Musical Instruments",
        "OfficeProducts": "Office Products",
        "HomeImprovement": "Home Improvement",
    }
    for old, new in replacements_cat.items():
        splitcategory["category"] = splitcategory["category"].str.replace(old, new)

    # Subcategory replacements
    replacements_sub = {
        "&": " & ",
        ",": ", ",
        "NetworkingDevices": "Networking Devices",
        "HomeTheater": "Home Theater",
        "HomeAudio": "Home Audio",
        "WearableTechnology": "Wearable Technology",
        "ExternalDevices": "External Devices",
        "DataStorage": "Data Storage",
        "GeneralPurposeBatteries": "General Purpose Batteries",
        "BatteryChargers": "Battery Chargers",
        "OfficePaperProducts": "Office Paper Products",
        "CraftMaterials": "Craft Materials",
        "OfficeElectronics": "Office Electronics",
        "PowerAccessories": "Power Accessories",
        "HomeAppliances": "Home Appliances",
        "AirQuality": "Air Quality",
        "HomeStorage": "Home Storage",
        "CarAccessories": "Car Accessories",
        "HomeMedicalSupplies": "Home Medical Supplies",
    }
    for old, new in replacements_sub.items():
        splitcategory["subcategory"] = splitcategory["subcategory"].str.replace(old, new)

    df = df.drop(columns="category")
    df["category"] = splitcategory["category"]
    df["subcategory"] = splitcategory["subcategory"]
    return df


def add_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``ranking`` column based on the ``rating`` value."""

    df = df.copy()
    ranking: list[str] = []

    for score in df["rating"]:
        if score <= 0.9:
            ranking.append("Muy Malo")
        elif score <= 1.9:
            ranking.append("Malo")
        elif score <= 2.9:
            ranking.append("Promedio")
        elif score <= 3.9:
            ranking.append("Bueno")
        elif score <= 4.9:
            ranking.append("Muy Bueno")
        elif score == 5.0:
            ranking.append("Excelente")
        else:
            ranking.append("Desconocido")

    df["ranking"] = pd.Categorical(ranking)
    return df


def prepare_reviewers(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe with reviewer information.

    The resulting dataframe contains the user id, user name, product name,
    category and subcategory.
    """

    df = df.copy()
    split_user_id = df["user_id"].str.split(",", expand=False)
    split_user_name = df["user_name"].str.split(",", expand=False)

    id_rows = split_user_id.explode().reset_index(drop=True)
    name_rows = split_user_name.explode().reset_index(drop=True)

    reviewers = pd.DataFrame({
        "user_id": id_rows,
        "user_name": name_rows,
        "product_name": df["product_name"],
        "category": df["category"],
        "subcategory": df["subcategory"],
    })

    reviewers = reviewers.dropna()
    return reviewers


__all__ = [
    "load_raw_data",
    "clean_prices",
    "split_categories",
    "add_ranking",
    "prepare_reviewers",
]

