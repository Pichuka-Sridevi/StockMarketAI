import tensorflow as tf

print("TensorFlow:", tf.__version__)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(5,)),
    tf.keras.layers.Dense(1)
])

model.summary()

print("TensorFlow is working!")
