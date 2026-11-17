"""Dashboard.  Start the API first, then:   streamlit run dashboard.py"""

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API = "http://127.0.0.1:8000"

st.set_page_config(page_title="Sales Forecast Dashboard", layout="wide")

col1, col2 = st.columns([3, 1])
with col1:
    st.title("📈 Sales Forecasting Dashboard")
    st.markdown("Forecast of the next 8 weeks of sales for each US state (weekly data)")

try:
    info = requests.get(f"{API}/model-info", timeout=5).json()
    states = requests.get(f"{API}/states", timeout=5).json()["states"]
    col2.success("API Running")
except Exception:
    col2.error("API not reachable")
    st.info("Start it in another terminal:  uvicorn api.app:app --reload")
    st.stop()

st.sidebar.header("Model Information")
st.sidebar.success(f"Model: {info['model']}")
st.sidebar.info(f"Forecast: next {info['forecast_weeks']} weeks")
st.sidebar.info(f"Data ends: {info['history_ends']}")

state = st.selectbox("Select State", states)

if st.button("Generate Forecast"):
    try:
        resp = requests.post(f"{API}/forecast", params={"state": state}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        st.stop()

    hist = pd.DataFrame(data["history"])
    fc = pd.DataFrame(data["forecast"])

    recent_avg = hist["sales"].tail(8).mean()
    fc_avg = fc["sales"].mean()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Average Forecast / week", f"{fc_avg:,.0f}")
    m2.metric("Highest week", f"{fc['sales'].max():,.0f}")
    m3.metric("Lowest week", f"{fc['sales'].min():,.0f}")
    m4.metric("vs last 8 weeks", f"{(fc_avg / recent_avg - 1) * 100:+.1f}%")

    st.subheader("Last 52 weeks and forecast")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=hist["date"], y=hist["sales"], name="Actual", mode="lines")
    )
    # join the two lines so the chart has no gap
    join = pd.concat([hist.tail(1).rename(columns={"sales": "sales"}), fc])
    fig.add_trace(
        go.Scatter(
            x=join["date"],
            y=join["sales"],
            name="Forecast",
            mode="lines+markers",
            line=dict(dash="dash"),
        )
    )
    fig.update_layout(
        title=f"{data['state']} - weekly sales",
        xaxis_title="Week",
        yaxis_title="Sales",
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast table")
    st.dataframe(
        fc.rename(columns={"date": "Week (Sunday)", "sales": "Forecast"}),
        use_container_width=True,
    )
