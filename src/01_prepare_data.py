"""Step 1: fix dates, build weekly features per state.

Run from the project root:  python src/01_prepare_data.py
"""

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "Forecasting Case- Study.xlsx"
OUT = ROOT / "data" / "processed" / "weekly_features.csv"

FEATURES = [
    "lag_1",
    "lag_2",
    "lag_4",
    "lag_8",
    "rolling_mean_4",
    "rolling_mean_8",
    "rolling_std_4",
    "week_of_year",
    "month",
    "quarter",
    "year",
    "has_holiday",
]


def fix_date(x):
    """The Excel column mixes real dates (day/month swapped) and text 'dd-mm-yyyy'."""
    if isinstance(x, str):
        return pd.to_datetime(x, format="%d-%m-%Y")
    return pd.Timestamp(year=x.year, month=x.day, day=x.month)


_HOLIDAYS = USFederalHolidayCalendar().holidays(start="2019-01-01", end="2024-12-31")


def has_holiday(week_end):
    """1 if any day in the 7 days ending on week_end is a US federal holiday."""
    days = pd.date_range(end=week_end, periods=7)
    return int(days.isin(_HOLIDAYS).any())


def build_features(df):
    """df: columns State, Date, Sales (weekly, sorted). Adds lag/rolling/calendar columns."""
    df = df.sort_values(["State", "Date"]).copy()
    g = df.groupby("State")["Sales"]
    for k in (1, 2, 4, 8):
        df[f"lag_{k}"] = g.shift(k)
    # shift(1) first so a row never "sees" its own sales
    df["rolling_mean_4"] = g.transform(lambda x: x.shift(1).rolling(4).mean())
    df["rolling_mean_8"] = g.transform(lambda x: x.shift(1).rolling(8).mean())
    df["rolling_std_4"] = g.transform(lambda x: x.shift(1).rolling(4).std())
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["month"] = df["Date"].dt.month
    df["quarter"] = df["Date"].dt.quarter
    df["year"] = df["Date"].dt.year
    df["has_holiday"] = df["Date"].map(has_holiday)
    return df


if __name__ == "__main__":
    raw = pd.read_excel(RAW)
    raw["Date"] = raw["Date"].map(fix_date)
    df = raw.rename(columns={"Total": "Sales"})[["State", "Date", "Sales"]]

    # sanity checks
    assert df.duplicated(["State", "Date"]).sum() == 0
    assert (
        df["Date"].drop_duplicates().sort_values().diff().dropna()
        == pd.Timedelta(days=7)
    ).all()

    # full clean weekly history (used by the model scripts)
    df.sort_values(["State", "Date"]).to_csv(
        ROOT / "data" / "processed" / "weekly_clean.csv", index=False
    )

    full = build_features(df)
    out = full.dropna()
    out.to_csv(OUT, index=False)
    print(
        f"states: {df.State.nunique()}  weeks per state: {df.groupby('State').size().iloc[0]}"
    )
    print(f"dates: {df.Date.min().date()} -> {df.Date.max().date()}")
    print(
        f"rows kept after dropping warm-up weeks: {len(out)} -> {OUT.relative_to(ROOT)}"
    )
