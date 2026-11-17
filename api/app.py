"""Forecast API.

Start from the project root:   uvicorn api.app:app --reload
Docs page (auto-made):         http://127.0.0.1:8000/docs
"""

import sys
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from common import ROOT, HORIZON, load_clean, forecast_state  # noqa: E402

app = FastAPI(title="Sales Forecast API")

model = joblib.load(ROOT / "models" / "final_model.pkl")
data = load_clean()
STATES = sorted(data["State"].unique().tolist())


def find_state(name: str) -> str:
    """Accept 'texas', 'TEXAS', ' Texas ' ... and return the proper name."""
    for s in STATES:
        if s.lower() == name.strip().lower():
            return s
    raise HTTPException(
        status_code=404,
        detail=f"Unknown state '{name}'. Use /states to see valid names.",
    )


@app.get("/")
def home():
    return {"message": "Forecast API Running"}


@app.get("/model-info")
def model_info():
    return {
        "model": "XGBoost (one model for all states)",
        "data_frequency": "weekly",
        "forecast_weeks": HORIZON,
        "history_ends": str(data["Date"].max().date()),
        "states": len(STATES),
    }


@app.get("/states")
def states():
    return {"states": STATES}


@app.post("/forecast")
def forecast(state: str):
    state = find_state(state)
    g = data[data["State"] == state]

    preds = forecast_state(model, g["Sales"].to_numpy(), g["Date"].iloc[-1], HORIZON)

    return {
        "state": state,
        "history": [
            {"date": str(d.date()), "sales": float(s)}
            for d, s in zip(g["Date"].tail(52), g["Sales"].tail(52))
        ],
        "forecast": [{"date": str(d.date()), "sales": float(p)} for d, p in preds],
    }
