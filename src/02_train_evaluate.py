"""Step 2: honest evaluation.

For every state: hide the last 8 weeks, train on the rest, forecast those 8 weeks
(recursively, like real use) and compare with what really happened.
Also compares against two simple baselines so we can tell if the model adds value.

Run from the project root:  python src/02_train_evaluate.py
"""

import numpy as np
import pandas as pd

from common import ROOT, HORIZON, load_clean, train_model, forecast_state

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def mape(actual, pred):
    actual, pred = np.asarray(actual), np.asarray(pred)
    return float(np.mean(np.abs((actual - pred) / actual)) * 100)


if __name__ == "__main__":
    df = load_clean()
    train_df = df[
        df.groupby("State").cumcount(ascending=False) >= HORIZON
    ]  # drop last 8 weeks per state
    print(
        f"weeks of data used for training: {len(train_df)}  (last {HORIZON} weeks of each state hidden)"
    )

    model = train_model(train_df)

    rows = []
    for state, g in df.groupby("State"):
        sales = g["Sales"].to_numpy()
        hist, actual = sales[:-HORIZON], sales[-HORIZON:]
        last_date = g["Date"].iloc[-HORIZON - 1]

        pred = np.array([p for _, p in forecast_state(model, hist, last_date)])
        naive = np.repeat(hist[-1], HORIZON)  # "same as last week"
        seasonal = hist[-52:][:HORIZON]  # "same weeks last year"

        rows.append(
            {
                "State": state,
                "MAPE_model": mape(actual, pred),
                "MAPE_naive": mape(actual, naive),
                "MAPE_last_year": mape(actual, seasonal),
                "MAE_model": float(np.mean(np.abs(actual - pred))),
            }
        )

    res = pd.DataFrame(rows).sort_values("MAPE_model")
    res.to_csv(OUT / "evaluation_by_state.csv", index=False)

    summary = (
        res[["MAPE_model", "MAPE_naive", "MAPE_last_year"]]
        .agg(["mean", "median"])
        .round(2)
    )
    summary.to_csv(OUT / "evaluation_summary.csv")
    print("\nAverage error across the 43 states (MAPE %, lower is better):")
    print(summary.to_string())
    print(
        "\nBest 3 states:\n",
        res.head(3)[["State", "MAPE_model"]].round(2).to_string(index=False),
    )
    print(
        "\nWorst 3 states:\n",
        res.tail(3)[["State", "MAPE_model"]].round(2).to_string(index=False),
    )
