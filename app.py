
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import subprocess
import sys

from pathlib import Path

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="StockSense AI",
    page_icon="📈",
    layout="wide"
)

# ==========================================
# 2. CUSTOM DARK THEME
# ==========================================

st.markdown("""
<style>
    .stApp {
        background-color: #0E1621;
        color: #E8EEF5;
    }

    h1, h2, h3 {
        color: #E8EEF5 !important;
        font-family: 'Segoe UI', sans-serif;
    }

    [data-testid="stMetric"] {
        background-color: #172434;
        padding: 22px;
        border-radius: 12px;
        border: 1px solid #26394D;
    }

    [data-testid="stMetricLabel"] {
        color: #A8B8C8;
    }

    [data-testid="stMetricValue"] {
        color: #75B3A0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. DASHBOARD HEADER
# ==========================================

st.title("📈 StockSense AI")
st.caption(
    "AI-Powered US Stock Market Prediction Dashboard"
)

st.divider()

# ==========================================
# 4. STOCKS AND MODEL FEATURES
# ==========================================

stocks = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL"]

features = [
    "Daily_Return",
    "Volatility_20",
    "RSI_14",
    "BB_Width",
    "Volume_Change",
    "Momentum_5D",
    "MA10_Ratio",
    "MA50_Ratio"
]

# ==========================================
# 5. MARKET DATA REFRESH
# ==========================================

st.subheader("Market Data Management")

if st.button(
    "🔄 Refresh Market Data",
    key="refresh_button"
):

    with st.spinner(
        "Downloading market data and updating models..."
    ):

        try:
            result = subprocess.run(
                [sys.executable, "refresh_data.py"],
                capture_output=True,
                text=True,
                check=True,
                timeout=300
            )

            # Store refresh results across reruns
            st.session_state["refresh_log"] = result.stdout
            st.session_state["refresh_success"] = True

            st.rerun()

        except subprocess.TimeoutExpired:
            st.session_state["refresh_success"] = False
            st.error(
                "Refresh timed out. Please try again."
            )

        except subprocess.CalledProcessError as e:
            st.session_state["refresh_success"] = False
            st.error("Market data refresh failed.")
            st.code(e.stderr or e.stdout)

# Display refresh confirmation
if st.session_state.get("refresh_success", False):

    st.success(
        "Market data and models refreshed successfully!"
    )

    with st.expander("View Refresh Details"):
        st.code(
            st.session_state.get("refresh_log", ""),
            language="text"
        )

    st.caption(
        "Refresh completed during this dashboard session."
    )

st.divider()

# ==========================================
# 6. STOCK SELECTION
# ==========================================

stock = st.selectbox(
    "Select a stock",
    stocks
)

data_path = Path(
    f"data/features/{stock}_features.csv"
)

model_path = Path(
    f"models/{stock}_final.pkl"
)

if not data_path.exists() or not model_path.exists():
    st.error(
        "Stock data or trained model not found. "
        "Please refresh the market data."
    )
    st.stop()

# ==========================================
# 7. LOAD STOCK DATA AND MODEL
# ==========================================

try:
    df = pd.read_csv(
        data_path,
        parse_dates=["Date"]
    )

    df = df.sort_values("Date")

    model = joblib.load(model_path)

    latest = df.dropna(
        subset=features
    ).iloc[[-1]]

except Exception as e:
    st.error(f"Unable to load stock data: {e}")
    st.stop()

# ==========================================
# 8. GENERATE STOCK PREDICTION
# ==========================================

probability_up = model.predict_proba(
    latest[features]
)[0, 1]

prediction = (
    "UP"
    if probability_up >= 0.5
    else "DOWN/FLAT"
)

# ==========================================
# 9. DASHBOARD METRICS
# ==========================================

st.subheader(f"{stock} — Stock Overview")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Latest Closing Price",
    f"${latest['Close'].iloc[0]:,.2f}"
)

col2.metric(
    "5-Day Prediction",
    prediction
)

col3.metric(
    "Model UP Score",
    f"{probability_up:.2%}"
)

# ==========================================
# 10. DATA FRESHNESS INFORMATION
# ==========================================

latest_date = pd.Timestamp(
    latest["Date"].iloc[0]
)

today = pd.Timestamp.now(
    tz="America/New_York"
).tz_localize(None)

days_old = (
    today.normalize()
    - latest_date.normalize()
).days

st.caption(
    f"Latest market data: {latest_date:%d %B %Y} | "
    "Prediction horizon: 5 trading days"
)

if days_old > 4:

    st.warning(
        f"Market data is {days_old} calendar days old. "
        "Refresh the dataset before relying on "
        "the displayed model output."
    )

else:

    st.success(
        "Market data is recent. "
        "Model predictions remain experimental."
    )

# ==========================================
# 11. INTERACTIVE CANDLESTICK CHART
# ==========================================

st.subheader(
    f"{stock} — Market Price Analysis"
)

chart_data = df.tail(180).copy()

fig = go.Figure()

# Candlestick prices
fig.add_trace(
    go.Candlestick(
        x=chart_data["Date"],
        open=chart_data["Open"],
        high=chart_data["High"],
        low=chart_data["Low"],
        close=chart_data["Close"],
        name="Price",
        increasing_line_color="#75B3A0",
        decreasing_line_color="#E57373"
    )
)

# 10-day moving average
fig.add_trace(
    go.Scatter(
        x=chart_data["Date"],
        y=chart_data["MA_10"],
        mode="lines",
        name="MA 10",
        line=dict(
            color="#91B9D0",
            width=1.5
        )
    )
)

# 50-day moving average
fig.add_trace(
    go.Scatter(
        x=chart_data["Date"],
        y=chart_data["MA_50"],
        mode="lines",
        name="MA 50",
        line=dict(
            color="#F2C879",
            width=1.5
        )
    )
)

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0E1621",
    plot_bgcolor="#0E1621",
    font=dict(color="#E8EEF5"),
    height=500,
    xaxis_title="Trading Date",
    yaxis_title="Stock Price (USD)",
    xaxis_rangeslider_visible=False,
    hovermode="x unified",
    legend=dict(
        orientation="h",
        y=1.08
    ),
    margin=dict(
        l=20,
        r=20,
        t=50,
        b=20
    )
)

fig.update_xaxes(
    rangeslider_visible=False,
    rangebreaks=[
        dict(bounds=["sat", "mon"])
    ]
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================
# 12. PREDICTION ANALYSIS
# ==========================================

st.subheader("Prediction Analysis")

st.write(
    f"The model currently predicts **{prediction}** "
    f"for **{stock}** over the next five trading days."
)

st.progress(float(probability_up))

st.caption(
    f"Model UP score: {probability_up:.2%}"
)

# ==========================================
# 13. HISTORICAL MODEL PERFORMANCE
# ==========================================

st.subheader("Historical Model Performance")

comparison_path = Path(
    "data/model_comparison.csv"
)

if comparison_path.exists():

    comparison = pd.read_csv(
        comparison_path
    )

    stock_results = comparison[
        comparison["Stock"] == stock
    ]

    if not stock_results.empty:

        stock_result = stock_results.iloc[0]

        col1, col2 = st.columns(2)

        col1.metric(
            "Walk-Forward Model Accuracy",
            f"{stock_result['Model_Accuracy']:.2f}%"
        )

        col2.metric(
            "Always-UP Baseline",
            f"{stock_result['Baseline_Accuracy']:.2f}%"
        )

        if (
            stock_result["Model_Accuracy"]
            < stock_result["Baseline_Accuracy"]
        ):

            st.info(
                "The model currently underperforms "
                "the historical Always-UP baseline."
            )

    else:
        st.info(
            "No performance results found for this stock."
        )

else:
    st.info(
        "Model comparison data is not available."
    )

# ==========================================
# 14. RISK DISCLAIMER
# ==========================================

st.divider()

st.warning(
    "StockSense AI is an experimental machine "
    "learning project. Predictions are not "
    "calibrated probabilities and historical "
    "testing has not demonstrated outperformance "
    "against a simple baseline. "
    "This dashboard is for educational and "
    "research purposes only, not financial advice."
)

st.caption(
    "StockSense AI | Machine Learning "
    "Stock Market Prediction Project"
)
