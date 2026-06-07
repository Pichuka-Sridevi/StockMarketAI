import numpy as np
import tensorflow as tf

X = np.random.rand(1000, 20, 13)
y = np.random.randint(0, 2, 1000)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(20,13)),
    tf.keras.layers.LSTM(16),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X,
    y,
    epochs=1,
    batch_size=32
)

print("Training Successful!")