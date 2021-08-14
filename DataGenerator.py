import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from typing import Tuple


class DataGenerator(keras.utils.Sequence):

    def __init__(self,
                 df: pd.DataFrame,
                 shape: Tuple[int, int] = (50, 13),
                 num_classes: int = 26,
                 batch_size: int = 32,
                 shuffle: bool = True):
        self.df = df
        self.shape = shape
        self.list_ids = self.df.index.tolist()
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        self.index = np.arange(len(self.list_ids))
        if self.shuffle:
            np.random.shuffle(self.index)

    def __getitem__(self, index):
        # generate indices of the batch
        indexes = self.index[index * self.batch_size:(index+1) * self.batch_size]

        # find list of ids
        batch_ids = [self.list_ids[k] for k in indexes]

        X, y = self.__get_data(batch_ids)
        return X, y

    def __len__(self):
        return len(self.list_ids) // self.batch_size

    def __get_data(self, batch_ids):
        X = np.empty((self.batch_size, *self.shape))
        y = np.empty(self.batch_size, dtype=int)

        for i, ID in enumerate(batch_ids):
            data = pd.read_csv(self.df['data_path'].iloc[ID])
            #data.drop(columns=['Unnamed: 0'], inplace=True)
            X[i,] = data.values
            y[i] = ord(self.df['label'].iloc[ID].upper()) - 65
        return X, tf.one_hot(y, depth=26)
