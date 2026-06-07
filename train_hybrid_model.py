import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ==========================
# SPEED CONFIG (CPU OPTIMIZED)
# ==========================
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

# ==========================
# LOAD DATA
# ==========================
df = pd.read_csv("multi_stock_dataset.csv")
print("Dataset Shape:", df.shape)

# ==========================
# CLEAN DATA
# ==========================
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna().reset_index(drop=True)

# ==========================
# ENCODE STOCK SYMBOL
# ==========================
encoder = LabelEncoder()
df["Stock"] = encoder.fit_transform(df["Stock"])

# ==========================
# FEATURES
# ==========================
features = [
    "Adj Close", "Close", "High", "Low", "Open", "Volume",
    "SMA20", "EMA20", "RSI", "MACD",
    "BB_High", "BB_Low", "Stock"
]

X = df[features].values
y = df["Target"].values

# ==========================
# SCALE
# ==========================
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# ==========================
# SEQUENCES
# ==========================
SEQ_LEN = 20

X_seq, y_seq = [], []

for i in range(SEQ_LEN, len(X)):
    X_seq.append(X[i-SEQ_LEN:i])
    y_seq.append(y[i])

X_seq = np.array(X_seq, dtype=np.float32)
y_seq = np.array(y_seq, dtype=np.float32)

print("Sequence Shape:", X_seq.shape)

# ==========================
# TRAIN TEST SPLIT
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X_seq,
    y_seq,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# ==========================
# FAST MODEL (LSTM ONLY)
# ==========================
model = Sequential([
    LSTM(32, return_sequences=True, input_shape=(SEQ_LEN, X_train.shape[2])),
    Dropout(0.2),

    LSTM(16),
    Dropout(0.2),

    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================
# TRAINING (FAST)
# ==========================
print("\nStarting Training...\n")

history = model.fit(
    X_train,
    y_train,
    epochs=3,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

# ==========================
# EVALUATION
# ==========================
loss, acc = model.evaluate(X_test, y_test, verbose=1)

print("\nTest Accuracy:", acc)

# ==========================
# SAVE MODEL
# ==========================
model.save("fast_stock_model.keras")

print("\nModel Saved Successfully!")