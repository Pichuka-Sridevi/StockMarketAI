import yfinance as yf
import pandas as pd
import ta

stocks = [
    "AAPL",
    "TSLA",
    "MSFT",
    "GOOGL",
    "AMZN",
    "IBM",
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS"
]

all_data = []

for symbol in stocks:

    print(f"Downloading {symbol}...")

    data = yf.download(
        symbol,
        period="5y",
        auto_adjust=False,
        progress=False
    )

    if len(data) == 0:
        continue

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

    bb = ta.volatility.BollingerBands(
        data["Close"],
        window=20
    )

    data["BB_High"] = bb.bollinger_hband()
    data["BB_Low"] = bb.bollinger_lband()

    data["Target"] = (
        data["Close"].shift(-1) > data["Close"]
    ).astype(int)

    data["Stock"] = symbol

    data = data.dropna()

    all_data.append(data)

final_df = pd.concat(all_data)

final_df.to_csv(
    "multi_stock_dataset.csv",
    index=False
)

print("\nDataset Created Successfully")
print(final_df.shape)