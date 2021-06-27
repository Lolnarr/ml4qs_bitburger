import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# https://www.tensorflow.org/guide/keras/rnn

# DATA_PATH = 'split_data'
DATA_PATH = 'git_data/split_data'
LETTER = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
          'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

def load_data(path) -> Tuple[np.ndarray, np.ndarray]:
    label = []
    data = []
    for foldername in sorted(os.listdir(path)):
        for filename in sorted(os.listdir(f'{path}/{foldername}')):
            df = pd.read_csv(f'{path}/{foldername}/{filename}', index_col=0)
            df.drop(columns=['id'], inplace=True)
            df.drop(columns=['time delta'], inplace=True)
            # df.drop(columns=['time'], inplace=True)
            array = df.to_numpy()
            sum = 0
            for item in array:
                sum += item
            avg = sum / len(array)
            fill_len = 200 - array.shape[0]
            # fill_len = 1000 - array.shape[0]
            full_array = np.full((fill_len, array.shape[1]), avg)
            array = np.concatenate((array, full_array))
            data.append(array)
            label.append(ord(foldername.upper())-65)
    data = np.stack(data, axis=0)
    label = np.array(label)
    return data, label


def build_model(input_shape: Tuple[int, int], num_classes: int) -> keras.Sequential:

    model = keras.Sequential(layers=[
        keras.layers.InputLayer(input_shape=input_shape),
        keras.layers.SimpleRNN(units=50, activation='relu', return_sequences=True),
        keras.layers.SimpleRNN(units=50, activation='relu', return_sequences=True),
        keras.layers.SimpleRNN(units=50, activation='relu', return_sequences=True),
        keras.layers.SimpleRNN(units=50, activation='relu', return_sequences=True),
        keras.layers.SimpleRNN(units=50, activation='relu', return_sequences=False),
        keras.layers.Dense(num_classes, activation=keras.activations.softmax)
    ])
    print(model.summary())
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    return model


def plot_confusion_matrix(labels, predictions, label_names):
    confusion_matrix = tf.math.confusion_matrix(labels, predictions)
    plt.figure()
    sns.heatmap(confusion_matrix, xticklabels=label_names, yticklabels=label_names, annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()
    pass


def main():
    x, y = load_data(DATA_PATH)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    y_train = tf.one_hot(y_train, depth=26)
    #y_test = tf.one_hot(y_test, depth=26)
    model = build_model((200, 13), 26)
    # model = build_model((1000, 6), 26)
    model.fit(x=x_train, y=y_train, epochs=10, batch_size=250, shuffle=True)
    y_predicted = np.argmax(model.predict(x=x_test), axis=1)
    plot_confusion_matrix(y_test, y_predicted, LETTER)


if __name__ == '__main__':
    main()
