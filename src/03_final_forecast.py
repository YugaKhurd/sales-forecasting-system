"""Step 3: train on ALL data and forecast the next 8 weeks for every state.

Run from the project root:  python src/03_final_forecast.py
Creates:
  models/final_model.pkl            the trained model (used by the API)
  outputs/forecast_8weeks.csv       State, Date, Forecast for every state
"""

import joblib
import pandas as pd

from common import ROOT, HORIZON, load_clean, train_model, forecast_state

if __name__ == "__main__":
    df = load_clean()
    model = train_model(df)

    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "outputs").mkdir(exist_ok=True)
    joblib.dump(model, ROOT / "models" / "final_model.pkl")

    rows = []
    for state, g in df.groupby("State"):
        for date, pred in forecast_state(
            model, g["Sales"].to_numpy(), g["Date"].iloc[-1], HORIZON
        ):
            rows.append(
                {"State": state, "Date": date.date(), "Forecast": round(pred, 2)}
            )

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "outputs" / "forecast_8weeks.csv", index=False)

    print(
        f"model saved, {out.State.nunique()} states x {HORIZON} weeks = {len(out)} rows"
    )
    print(f"forecast period: {out.Date.min()} -> {out.Date.max()}")
    print(out[out.State == "Texas"].to_string(index=False))
