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


DATA_PATH = 'git_data/split_data'
TRAIN_PATH = 'DATA/ahrs_data/training'
TEST_PATH = 'DATA/ahrs_data/test'
VAL_PATH = 'DATA/ahrs_data/validation'
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


def plot_history(history: keras.callbacks.History):
    fig, axs = plt.subplots(2)
    fig.suptitle('Training History', fontsize=16)
    axs[0].plot(history.epoch, history.history['loss'], history.history['val_loss'])
    axs[0].set(title='Loss', xlabel='Epoch', ylabel='Loss')
    axs[0].legend(['loss', 'val_loss'])
    axs[1].plot(history.epoch, history.history['accuracy'], history.history['val_accuracy'])
    axs[1].set(title='Accuracy', xlabel='Epoch', ylabel='Accuracy')
    axs[1].legend(['accuracy', 'val_accuracy'])
    plt.show()


def main():
    SHAPE_X = 100
    SHAPE_Y = 10

    train_generator = DataGenerator(get_path_df(TRAIN_PATH), shape=(SHAPE_X, SHAPE_Y), batch_size=32)
    val_generator = DataGenerator(get_path_df(VAL_PATH), shape=(SHAPE_X, SHAPE_Y), batch_size=32)
    test_generator = DataGenerator(get_path_df(TEST_PATH), shape=(SHAPE_X, SHAPE_Y), batch_size=32)

    model = build_model((SHAPE_X, SHAPE_Y), 26)
    history = model.fit(train_generator, epochs=50, validation_data=val_generator)
    loss, acc = model.evaluate(test_generator)
    print('test loss:', loss)
    print('test accuracy:', acc)
    model.save(filepath='saved_models/100units_32batch_50epochs_ahrs/model.h5', overwrite=True)

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

    plot_history(history)


if __name__ == '__main__':
    main()
