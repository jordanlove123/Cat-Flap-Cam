import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1,784).astype("float32") / 255.0
x_test = x_test.reshape(-1,784).astype("float32") / 255.0

model = keras.Sequential(
    [
        keras.Dense(64, activation='relu'),
        keras.Dense(32, activation='relu'),
        keras.Dense(10)
    ]    
)

model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    optimizer = keras.optimizers.Adam(lr = 0.001),
    metrics = ["accuracy"]
)

model.fit(x_train, y_train, batch_size = 32, epochs = 5, verbose = 2)
model.evaluate(x_test, y_test, batch_size = 32, verbose = 2)

def sigmoid(num):
    return 1 / (1+(np.e**-num))

if __name__ == "__main__":
    print(x_train.shape)
    print(y_train.shape)