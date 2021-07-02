import os
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from typing import List, Tuple

# Set constants
TRAIN_PATH = pathlib.Path('UCI HAR Dataset/train')
TEST_PATH = pathlib.Path('UCI HAR Dataset/test')
DOWNLOAD_LINK = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip'
TEST_SIZE = 0.3
BATCH_SIZE = 512
EPOCHS = 20
LABELS = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]


def load_data_from_directory(data_path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray]:
    print(f'Loading data from directory "{data_path}"')
    # Load signals / features
    signals = []
    signals_path = os.path.join(data_path, 'Inertial Signals')
    for signal_name in tqdm(os.listdir(signals_path)):
        signal_data = pd.read_csv(os.path.join(signals_path, signal_name), sep='\s+')
        signals.append(signal_data.to_numpy(dtype=np.float32))
    # Load labels and shift them by 1 to have easier indexing when checking description in LABELS
    labels_filename = 'y_train.txt' if os.path.exists(os.path.join(data_path, 'y_train.txt')) else 'y_test.txt'
    labels = pd.read_csv(os.path.join(data_path, labels_filename)).to_numpy().squeeze() - 1
    # Return transposed features and labels as Numpy arrays
    return np.transpose(np.array(signals), (1, 2, 0)), labels


def visualize_class_instances(classes: List[str], X: np.ndarray, y: np.ndarray):
    FEATURE_NAMES = ['Body_Acc_X', 'Body_Acc_Y', 'Body_Acc_Z',
                     'Body_Gyro_X', 'Body_Gyro_Y', 'Body_Gyro_Z',
                     'Total_Acc_X', 'Total_Acc_Y', 'Total_Acc_Z']
    fig, axs = plt.subplots(nrows=3, ncols=len(classes), sharex=True, sharey=False)
    fig.suptitle('Visualization of different Classes', fontsize=16)
    x = np.arange(128)
    for index, cls in enumerate(classes):
        # Set class as label for top subplot of each column
        axs[0][index].set(title=cls)
        # Get random instance from training data with corresponding class
        instance_index = np.random.choice(np.argwhere(y == LABELS.index(cls)).flatten())
        instance = X[instance_index]
        for sublot in [0, 1, 2]:
            axs[sublot][index].plot(x, instance[:, sublot * 3])
            axs[sublot][index].plot(x, instance[:, sublot * 3 + 1])
            axs[sublot][index].plot(x, instance[:, sublot * 3 + 2])
            axs[sublot][index].legend(FEATURE_NAMES[sublot * 3:sublot * 3 + 3])
    plt.show()


def build_model(input_shape: Tuple[int], num_classes: int, model_type: str = 'simple') -> keras.Sequential:
    model_type = model_type if model_type in ['simple', 'lstm', 'gru'] else 'simple'
    if model_type == 'simple':
        rnn_type = keras.layers.SimpleRNN
    elif model_type == 'lstm':
        rnn_type = keras.layers.LSTM
    elif model_type == 'gru':
        rnn_type = keras.layers.GRU

    model = keras.Sequential(layers=[
        keras.layers.InputLayer(input_shape=input_shape),
        rnn_type(units=128, activation='relu', return_sequences=True),
        rnn_type(units=128, activation='relu', return_sequences=False),
        keras.layers.Dense(num_classes, activation=keras.activations.softmax)
    ])
    print(model.summary())
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    return model


def plot_training_history(history: keras.callbacks.History):
    fig, axs = plt.subplots(2)
    fig.suptitle('Training History', fontsize=16)
    axs[0].plot(history.epoch, history.history['loss'], history.history['val_loss'])
    axs[0].set(title='Loss', xlabel='Epoch', ylabel='Loss')
    axs[0].legend(['loss', 'val_loss'])
    axs[1].plot(history.epoch, history.history['accuracy'], history.history['val_accuracy'])
    axs[1].set(title='Accuracy', xlabel='Epoch', ylabel='Accuracy')
    axs[1].legend(['accuracy', 'val_accuracy'])
    plt.show()


def plot_confusion_matrix(y_true: tf.Tensor, y_predicted: tf.Tensor, label_names: List[str]):
    confusion_matrix = tf.math.confusion_matrix(labels=y_true, predictions=y_predicted)
    plt.figure()
    sns.heatmap(confusion_matrix, xticklabels=label_names, yticklabels=label_names, annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()


def main():
    # Load train and test data from corresponding directories
    X_train, y_train = load_data_from_directory(data_path=TRAIN_PATH)
    X_test, y_test = load_data_from_directory(data_path=TEST_PATH)

    print("X train:")
    print(X_train)

    # Visualize class instances
    visualize_class_instances(classes=['WALKING_UPSTAIRS', 'WALKING'], X=X_train, y=y_train)

    # Encode the labels using One-Hot-Encoding
    y_train_encoded = tf.one_hot(indices=y_train, depth=6)

    # Determine input shape and number of classes to build model
    input_shape = X_train[0].shape
    print(input_shape)
    num_classes = len(LABELS)
    model = build_model(input_shape=input_shape, num_classes=num_classes, model_type='simple')

    # Train model using validation split and plot the history
    history = model.fit(x=X_train, y=y_train_encoded, validation_split=TEST_SIZE, epochs=EPOCHS, batch_size=BATCH_SIZE)
    plot_training_history(history=history)

    # Let the model predict the testset and plot confusion matrix
    y_predicted = np.argmax(model.predict(x=X_test), axis=1)
    plot_confusion_matrix(y_true=y_test, y_predicted=y_predicted, label_names=LABELS)

    # Save the model using tensorflow
    model.save(filepath='model_git_norm_2.h5', overwrite=True)


if __name__ == '__main__':
    main()