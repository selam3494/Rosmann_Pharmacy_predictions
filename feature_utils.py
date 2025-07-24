# feature_utils.py
import pandas as pd
from typing import Tuple, List
from path_utils import DATA_DIR     # <-- use shared constants

def feature_engineer(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Add derived columns and cast categoricals. Safe for any dataframe shaped like train/test."""
    if "Date" not in df.columns:
        raise ValueError("DataFrame must contain a 'Date' column")

    df["Date"] = pd.to_datetime(df["Date"])

    # Fill missing mandatory columns so KeyError never occurs
    defaults = {
        "StateHoliday": "0",
        "Promo2": 0,
        "Promo2SinceYear": df["Date"].dt.year,
        "Promo2SinceWeek": df["Date"].dt.isocalendar().week.astype(int),
        "PromoInterval": "0",
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    # Date parts
    df["Year"]       = df["Date"].dt.year
    df["Month"]      = df["Date"].dt.month
    df["Day"]        = df["Date"].dt.day
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)

    # Competition & Promo windows
    df["CompOpen"] = (
        12 * (df["Year"]  - df["CompetitionOpenSinceYear"].fillna(df["Year"]))
        +  (df["Month"] - df["CompetitionOpenSinceMonth"].fillna(df["Month"]))
    ).clip(lower=0)

    df["Promo2Open"] = (
        12 * (df["Year"] - df["Promo2SinceYear"].fillna(df["Year"])) * 4.348
        +  (df["WeekOfYear"] - df["Promo2SinceWeek"].fillna(df["WeekOfYear"]))
    ).clip(lower=0)

    # Categoricals
    cat_cols = [
        "Store", "DayOfWeek", "StateHoliday", "StoreType",
        "Assortment", "Promo", "Promo2", "SchoolHoliday", "PromoInterval"
    ]
    for c in cat_cols:
        df[c] = df[c].astype("category")

    return df, cat_cols
