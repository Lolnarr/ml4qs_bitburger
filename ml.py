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
from sklearn.metrics import confusion_matrix

from DataGenerator import DataGenerator

# https://www.tensorflow.org/guide/keras/rnn

# DATA_PATH = 'split_data'
DATA_PATH = 'split_data'
TRAIN_PATH = 'normalized_data/training'
TEST_PATH = 'normalized_data/test'
VAL_PATH = 'normalized_data/validation'
LETTER = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
          'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


def get_path_df(path: str) -> pd.DataFrame:
    data_dict = {'data_path': [], 'label': []}
    for folder_name in os.listdir(path):
        for file_name in os.listdir(f'{path}/{folder_name}'):
            data_dict['data_path'].append(f'{path}/{folder_name}/{file_name}')
            data_dict['label'].append(folder_name)
    return pd.DataFrame(data_dict)


def load_data(path) -> Tuple[np.ndarray, np.ndarray]:
    label = []
    data = []
    for foldername in sorted(os.listdir(path)):
        for filename in sorted(os.listdir(f'{path}/{foldername}')):
            df = pd.read_csv(f'{path}/{foldername}/{filename}', index_col=0)
            # df.drop(columns=['id'], inplace=True)
            # df.drop(columns=['time delta'], inplace=True)
            df.drop(columns=['time'], inplace=True)
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
        keras.layers.LSTM(units=100, return_sequences=True),
        keras.layers.LSTM(units=100, return_sequences=True),
        keras.layers.LSTM(units=100),
        keras.layers.Dense(num_classes, activation=keras.activations.softmax)
    ])
    """
    model = keras.Sequential(layers=[
        keras.layers.InputLayer(input_shape=input_shape),
        keras.layers.SimpleRNN(units=50, activation='relu', return_sequences=True),
        keras.layers.SimpleRNN(units=50, activation='relu', return_sequences=True),
        keras.layers.SimpleRNN(units=50, activation='relu', return_sequences=True),
        keras.layers.SimpleRNN(units=50, activation='relu', return_sequences=True),
        keras.layers.SimpleRNN(units=50, activation='relu', return_sequences=False),
        keras.layers.Dense(num_classes, activation=keras.activations.softmax)
    ])
    """
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
    train_generator = DataGenerator(get_path_df(TRAIN_PATH), shape=(500, 6), batch_size=32)
    val_generator = DataGenerator(get_path_df(VAL_PATH), shape=(500, 6), batch_size=32)
    test_generator = DataGenerator(get_path_df(TEST_PATH), shape=(500, 6), batch_size=32)

    # x, y = load_data(DATA_PATH)
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    # y_train = tf.one_hot(y_train, depth=26)
    # y_test = tf.one_hot(y_test, depth=26)
    model = build_model((500, 6), 26)
    # model = build_model((1000, 6), 26)
    model.fit(train_generator, epochs=30, validation_data=val_generator)
    # model.fit(train_generator, epochs=10, batch_size=128, shuffle=True)
    # y_predicted = model.predict(test_generator)
    # plot_confusion_matrix(test_generator.classes, y_predicted, LETTER)

    n_batches = len(test_generator)

    mat = confusion_matrix(
        np.concatenate([np.argmax(test_generator[i][1], axis=1) for i in range(n_batches)]),
        np.argmax(model.predict(test_generator, steps=n_batches), axis=1)
    )

    plt.figure()
    sns.heatmap(mat, xticklabels=[i.upper() for i in LETTER],
                yticklabels=[i.upper() for i in LETTER], annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()


if __name__ == '__main__':
    main()
