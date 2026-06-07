import tensorflow as tf
import joblib
import numpy as np
import streamlit as st
import yfinance as yf
import pandas as pd
import ta

from datetime import datetime

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Stock AI Dashboard",
    page_icon="📈",
    layout="wide"
)

# ==================================================
# CUSTOM CSS
# ==================================================

# ==================================================
# LOAD MODELS
# ==================================================
lstm_model = tf.keras.models.load_model(
    "lstm_model.keras"
)

xgb_model = joblib.load(
    "xgboost_model.pkl"
)

scaler = joblib.load(
    "scaler.pkl"
)

# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:

    st.title("📊 Stock AI Dashboard")

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "📈 Prediction",
            "📊 Technical Analysis",
            "📥 Reports",
            "ℹ️ About"
        ]
    )

    st.markdown("---")

    st.success("🤖 Hybrid AI")
    st.caption("LSTM + XGBoost")
    st.markdown("---")

    st.info("Hybrid Accuracy: 52.8%")

# ==================================================
# HEADER
# ==================================================
st.title("📈 Stock Market Trend Prediction")

st.success("AI Models Loaded")

# ==================================================
# STOCK INPUT
# ==================================================
symbol = st.text_input(
    "Enter Stock Symbol",
    "IBM"
)

# ==================================================
# COMPANY NAME
# ==================================================
try:

    INDEX_NAMES = {
        "^BSESN": "BSE SENSEX",
        "^NSEI": "NIFTY 50",
        "^DJI": "DOW JONES",
        "^GSPC": "S&P 500",
        "^IXIC": "NASDAQ"
    }

    if symbol.upper() in INDEX_NAMES:
        company_name = INDEX_NAMES[symbol.upper()]
    else:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        company_name = info.get(
            "longName",
            symbol.upper()
        )

except:
    company_name = symbol.upper()


st.markdown(f"""
<div style="
padding:20px;
background:white;
border-radius:20px;
box-shadow:0 4px 12px rgba(0,0,0,0.08);
text-align:center;
margin-bottom:15px;
">
<h1 style="color:#1e3a5f;">{company_name}</h1>
<p style="font-size:18px;">{symbol.upper()}</p>
</div>
""", unsafe_allow_html=True)


# ==================================================
# FETCH DATA
# ==================================================
if st.button("Fetch Data"):

    data = yf.download(
        symbol,
        period="5y",
        auto_adjust=False
    )

    if data.empty:
        st.error("No data found")
        st.stop()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # ==========================================
    # TECHNICAL INDICATORS
    # ==========================================
    data["SMA20"] = ta.trend.sma_indicator(
        data["Close"],
        window=20
    )

    data["EMA20"] = ta.trend.ema_indicator(
        data["Close"],
        window=20
    )

    data["RSI"] = ta.momentum.rsi(
        data["Close"],
        window=14
    )

    data["MACD"] = ta.trend.macd(
        data["Close"]
    )

    bollinger = ta.volatility.BollingerBands(
        data["Close"],
        window=20
    )

    data["BB_High"] = bollinger.bollinger_hband()
    data["BB_Low"] = bollinger.bollinger_lband()

    data["Stock"] = 0

    data = data.dropna()

    # ==========================================
    # FEATURES
    # ==========================================
    features = [
        "Adj Close",
        "Close",
        "High",
        "Low",
        "Open",
        "Volume",
        "SMA20",
        "EMA20",
        "RSI",
        "MACD",
        "BB_High",
        "BB_Low",
        "Stock"
    ]

    latest_data = data[features].tail(20)

    scaled_data = scaler.transform(
        latest_data
    )

    X_lstm = np.array([scaled_data])

    lstm_features = lstm_model.predict(
        X_lstm,
        verbose=0
    )

    X_last = scaled_data[-1].reshape(
        1,
        -1
    )

    hybrid_features = np.concatenate(
        [lstm_features, X_last],
        axis=1
    )

    xgb_pred = float(
        xgb_model.predict_proba(
            hybrid_features
        )[0][1]
    )

    hybrid_score = xgb_pred

    prediction_text = (
        "UP 🚀"
        if hybrid_score > 0.5
        else "DOWN 📉"
    )

    confidence = max(
        hybrid_score,
        1 - hybrid_score
    )

    current_price = float(
        data["Close"].iloc[-1]
    )

    previous_price = float(
        data["Close"].iloc[-2]
    )

    change_pct = (
        (current_price - previous_price)
        / previous_price
    ) * 100

    # ==================================================
    # DASHBOARD PAGE
    # ==================================================
    if page == "🏠 Dashboard":

        st.header("📊 Dashboard")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Price",
            f"${current_price:.2f}"
        )

        col2.metric(
            "Signal",
            prediction_text
        )

        col3.metric(
            "Confidence",
            f"{confidence:.1%}"
        )

        col4.metric(
            "Volume",
            f"{data['Volume'].iloc[-1]:,.0f}"
        )
    # ==================================================
    # PREDICTION PAGE
    # ==================================================
    elif page == "📈 Prediction":

        st.header("📈 AI Prediction")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Prediction",
            prediction_text
        )

        col2.metric(
            "Confidence",
            f"{confidence:.2%}"
        )

        col3.metric(
            "AI Score",
            f"{hybrid_score:.4f}"
        )

        if hybrid_score > 0.5:
            st.success("🟢 BUY SIGNAL")
        else:
            st.error("🔴 SELL SIGNAL")

        st.subheader("Confidence Level")

        st.write(
            f"### {confidence:.1%}"
        )

        st.progress(
            float(confidence)
        )

        st.write(
            f"XGBoost Probability: {xgb_pred:.4f}"
        )

    # ==================================================
    # TECHNICAL ANALYSIS
    # ==================================================
    elif page == "📊 Technical Analysis":

        st.header("📊 Technical Analysis")

        st.subheader("📈 Stock Price")
        st.line_chart(data["Close"])

        st.subheader("📊 RSI")
        st.line_chart(data["RSI"])

        st.subheader("📉 MACD")
        st.line_chart(data["MACD"])

        st.subheader("📋 Latest Data")
        st.dataframe(data.tail())

    # ==================================================
    # REPORTS
    # ==================================================
    elif page == "📥 Reports":

        st.header("📥 Reports")

        report = pd.DataFrame({
            "Date":[datetime.now()],
            "Symbol":[symbol],
            "Prediction":[prediction_text],
            "Confidence":[confidence],
            "AI Score":[hybrid_score]
        })

        csv = report.to_csv(
            index=False
        )

        st.download_button(
            label="📥 Download Prediction Report",
            data=csv,
            file_name=f"{symbol}_report.csv",
            mime="text/csv"
        )

        st.dataframe(report)

# ==================================================
# ABOUT PAGE
# ==================================================
if page == "ℹ️ About":

    st.header("ℹ️ About This Project")

    with st.expander("Model Details"):

        st.write("""
        LSTM captures market patterns
        from historical sequences.

        XGBoost learns market relationships
        from extracted features.

        Together they create a Hybrid
        AI Prediction System.
        """)

    st.info(
        "Hybrid Accuracy: 52.8%"
    )

    st.success(
        "Built using Streamlit, TensorFlow, XGBoost and Yahoo Finance."
    )
    st.markdown("---")

    st.caption(
        "📈 Stock AI Dashboard | TensorFlow | XGBoost | Streamlit"
    )