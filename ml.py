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

    for foldername in sorted(os.listdir(path)):
        for filename in sorted(os.listdir(f'{path}/{foldername}')):
            df = pd.read_csv(f'{path}/{foldername}/{filename}')
            array = df.to_numpy()
            fill_len = 1000 - array.shape[0]
            full_array = np.full((fill_len, array.shape[1]), 0)
            array = np.concatenate((array, full_array))
            print(array.shape)
            for row in array:
                print(row)



def main():
    load_data(DATA_PATH)


if __name__ == '__main__':
    main()
