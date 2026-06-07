import yfinance as yf
import pandas as pd
import ta

symbol = input("Enter Stock Symbol: ")

data = yf.download(
    symbol,
    period="5y",
    auto_adjust=False
)

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

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

# Target column

data["Target"] = (
    data["Close"].shift(-1) > data["Close"]
).astype(int)

data = data.dropna()

data.to_csv("dataset.csv")

print(data.head())

print("\nDataset Saved Successfully")