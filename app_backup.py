import tensorflow as tf
import joblib
import numpy as np
import streamlit as st
import yfinance as yf
import pandas as pd
import ta
lstm_model = tf.keras.models.load_model(
    "lstm_model.keras"
)

xgb_model = joblib.load(
    "xgboost_model.pkl"
)

scaler = joblib.load(
    "scaler.pkl"
)


st.title("Stock Market Trend Prediction")
lstm_model = tf.keras.models.load_model(
    "lstm_model.keras"
)

xgb_model = joblib.load(
    "xgboost_model.pkl"
)

st.success("AI Models Loaded")

symbol = st.text_input(
    "Enter Stock Symbol",
    "IBM"
)

if st.button("Fetch Data"):

    # Download stock data
    data = yf.download(
        symbol,
        period="5y",
        auto_adjust=False
    )

    # Fix MultiIndex columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Technical Indicators
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

    scaled_data = scaler.transform(latest_data)

    X_lstm = np.array([scaled_data])

    # Get 32 LSTM features
    lstm_features = lstm_model.predict(
        X_lstm,
        verbose=0
    )

    # Get last day's 13 features
    X_last = scaled_data[-1].reshape(1, -1)

    # Build 45-feature hybrid input
    hybrid_features = np.concatenate(
        [lstm_features, X_last],
        axis=1
    )

    # XGBoost prediction
    xgb_pred = float(
        xgb_model.predict_proba(
            hybrid_features
        )[0][1]
    )

    hybrid_score = xgb_pred

    st.subheader("📈 AI Prediction")

    if hybrid_score > 0.5:
        st.success(
            f"Prediction: UP 🚀 ({hybrid_score:.2%} confidence)"
        )
    else:
        st.error(
            f"Prediction: DOWN 📉 ({1-hybrid_score:.2%} confidence)"
        )

    st.write(f"XGBoost Probability: {xgb_pred:.4f}")
    st.write(f"Hybrid Score: {hybrid_score:.4f}")
    
    # Show latest data
    st.subheader("Latest Data")
    st.dataframe(data.tail())

    # Price Chart
    st.subheader("Stock Price Chart")
    st.line_chart(data["Close"])

    # RSI Chart
    st.subheader("RSI Chart")
    st.line_chart(data["RSI"])

    # MACD Chart
    st.subheader("MACD Chart")
    st.line_chart(data["MACD"])