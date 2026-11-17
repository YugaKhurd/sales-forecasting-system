# Sales Forecasting System

Forecasts the next **8 weeks of beverage sales for each of 43 US states**, and serves the forecasts through a REST API and an interactive dashboard.

## Data

- Source: `data/raw/Forecasting Case- Study.xlsx` — weekly sales (every Sunday) for 43 states, 6 Oct 2019 to 7 May 2023 (188 weeks per state).
- **Data issue found and fixed:** the Excel date column mixes real dates with day and month swapped and text dates like `31-01-2021`. They are repaired in `src/01_prepare_data.py` and checked (no duplicates, every gap exactly 7 days).

## Approach

1. **Prepare data** — repair dates, check the weekly series.
2. **Features** — sales 1, 2, 4 and 8 weeks ago, 4-week average and spread, week of year, month, quarter, US-holiday flag. Every value is divided by the state's recent 8-week average, so one model can learn from small and large states alike.
3. **Model** — a single XGBoost regressor trained on all states.
4. **Multi-week forecasting** — recursive: predict week 1, add it to the history, predict week 2, and so on, so each forecast week uses fresh features.
5. **Serving** — FastAPI endpoint + Streamlit dashboard.

## Results

Evaluated by hiding the last 8 weeks of every state, training on the rest and forecasting those 8 weeks. MAPE = average % error (lower is better), averaged over 43 states:

| Method                     | MAPE      |
| -------------------------- | --------- |
| **XGBoost (this project)** | **4.76%** |
| Same as last week (naive)  | 5.69%     |
| Same weeks last year       | 5.46%     |

Per-state results: `outputs/evaluation_by_state.csv`. Best states are around 2% error; the hardest (Michigan, Wyoming, Maine) are around 7%.

**Limitations:** the improvement over simple baselines is moderate; only 3.5 years of history; no outside information (promotions, weather, prices); one 8-week test window per state.

## Project structure

```
src/01_prepare_data.py     clean dates, build weekly data
src/common.py              features, training, recursive forecasting
src/02_train_evaluate.py   honest evaluation vs baselines
src/03_final_forecast.py   train on all data, forecast next 8 weeks
api/app.py                 FastAPI service
dashboard.py               Streamlit dashboard
notebooks/                 original exploratory work (see note below)
```

## How to run

```bash
pip install -r requirements.txt
python src/01_prepare_data.py
python src/02_train_evaluate.py      # optional: prints accuracy
python src/03_final_forecast.py      # trains the final model

# two terminals:
uvicorn api.app:app --reload         # API  -> http://127.0.0.1:8000/docs
streamlit run dashboard.py           # dashboard
```

## API

| Method | Path                    | Description                            |
| ------ | ----------------------- | -------------------------------------- |
| GET    | `/`                     | health check                           |
| GET    | `/model-info`           | model, data end date, number of states |
| GET    | `/states`               | valid state names                      |
| POST   | `/forecast?state=Texas` | last 52 weeks + next 8 weeks forecast  |

## Note on the notebooks

`notebooks/` holds the first version of the project. It treated the data as daily and misread the dates, so its results (including the SARIMA / Prophet / LSTM comparison) should not be trusted. The `src/` scripts replace it.
