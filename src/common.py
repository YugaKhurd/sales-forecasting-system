"""Shared code: feature creation, training, and multi-week forecasting.

The SAME function (row_features) is used for training and for forecasting,
so the model never sees features computed in a different way.

Idea: sales differ hugely between states (10M to 900M). So every lag/average is
divided by the state's recent average (the "scale"). The model learns
"what happens next relative to recent level", which works for all states.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "processed" / "weekly_clean.csv"

HORIZON = 8  # weeks to forecast
MIN_HISTORY = 8  # weeks needed before the first usable row

FEATURES = [
    "lag_1",
    "lag_2",
    "lag_4",
    "lag_8",
    "rolling_mean_4",
    "rolling_std_4",
    "week_of_year",
    "month",
    "quarter",
    "has_holiday",
]

_HOLIDAYS = USFederalHolidayCalendar().holidays(start="2018-01-01", end="2030-12-31")


def load_clean():
    df = pd.read_csv(CLEAN, parse_dates=["Date"])
    return df.sort_values(["State", "Date"]).reset_index(drop=True)


def row_features(history, date):
    """Features for the week ending on `date`, given all earlier weekly sales (oldest first).
    Returns (feature_dict, scale)."""
    h = np.asarray(history, dtype=float)
    scale = h[-8:].mean()
    days = pd.date_range(end=date, periods=7)
    feats = {
        "lag_1": h[-1] / scale,
        "lag_2": h[-2] / scale,
        "lag_4": h[-4] / scale,
        "lag_8": h[-8] / scale,
        "rolling_mean_4": h[-4:].mean() / scale,
        "rolling_std_4": h[-4:].std(ddof=1) / scale,
        "week_of_year": int(date.isocalendar()[1]),
        "month": date.month,
        "quarter": (date.month - 1) // 3 + 1,
        "has_holiday": int(days.isin(_HOLIDAYS).any()),
    }
    return feats, scale


def build_training_set(df):
    """One row per (state, week) that has at least 8 weeks of history before it.
    Target = sales / scale (so about 1.0 means 'same as recent level')."""
    rows, targets = [], []
    for _, g in df.groupby("State"):
        sales = g["Sales"].to_numpy()
        dates = g["Date"].tolist()
        for i in range(MIN_HISTORY, len(g)):
            feats, scale = row_features(sales[:i], dates[i])
            rows.append(feats)
            targets.append(sales[i] / scale)
    return pd.DataFrame(rows)[FEATURES], np.array(targets)


def make_model():
    from xgboost import XGBRegressor

    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )


def train_model(df):
    X, y = build_training_set(df)
    model = make_model()
    model.fit(X, y)
    return model


def forecast_state(model, history, last_date, weeks=HORIZON):
    """Recursive forecast: predict week 1, add it to the history, then predict week 2, etc."""
    hist = list(history)
    out = []
    date = pd.Timestamp(last_date)
    for _ in range(weeks):
        date = date + pd.Timedelta(days=7)
        feats, scale = row_features(hist, date)
        pred = float(model.predict(pd.DataFrame([feats])[FEATURES])[0]) * scale
        out.append((date, pred))
        hist.append(pred)
    return out
