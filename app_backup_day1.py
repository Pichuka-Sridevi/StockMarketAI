import streamlit as st
import yfinance as yf
import pandas as pd
import ta

st.title("Stock Market Trend Prediction")

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