import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple

# https://www.tensorflow.org/guide/keras/rnn

DATA_PATH = 'split_data'


def load_data(path) -> Tuple[np.ndarray, np.ndarray]:
    label = []
    data = []
    for foldername in sorted(os.listdir(path)):
        for filename in sorted(os.listdir(f'{path}/{foldername}')):
            df = pd.read_csv(f'{path}/{foldername}/{filename}', index_col=0)
            df.drop(columns=['time'], inplace=True)
            array = df.to_numpy()
            fill_len = 1000 - array.shape[0]
            full_array = np.full((fill_len, array.shape[1]), 0)
            array = np.concatenate((array, full_array))
            data.append(array)
            label.append(ord(foldername)-64)
    data = np.stack(data, axis=0)
    label = np.array(label)
    return data, label


def build_model(input_shape: Tuple[int, int], num_classes: int) -> keras.Sequential:

    model = keras.Sequential(layers=[
        keras.layers.InputLayer(input_shape=input_shape),
        keras.layers.SimpleRNN(units=500, activation='relu', return_sequences=True),
        keras.layers.SimpleRNN(units=500, activation='relu', return_sequences=False),
        keras.layers.Dense(num_classes, activation=keras.activations.softmax)
    ])
    print(model.summary())
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    return model


def main():
    x, y = load_data(DATA_PATH)
    y = tf.one_hot(y, depth=26)
    model = build_model((1000, 6), 26)
    model.fit(x=x, y=y, epochs=10, batch_size=64)


if __name__ == '__main__':
    main()
