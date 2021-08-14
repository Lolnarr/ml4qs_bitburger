from tensorflow import keras
from DataGenerator import DataGenerator
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

TEST_PATH = 'DATA/ahrs_data/predict_data'
LETTER = ['a', 'b', 'c', 'd', 'e', 'f','g', 'h', 'i', 'j', 'k', 'l', 'm',
          'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


def main():
    SHAPE_X = 100
    SHAPE_Y = 10

    model = keras.models.load_model('saved_models/100units_32batch_50epochs_ahrs/model.h5')

    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])

    test_generator = DataGenerator(get_path_df(TEST_PATH), shape=(SHAPE_X, SHAPE_Y), batch_size=1)

    model.evaluate(test_generator, batch_size=1, verbose=1)

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
    plt.savefig('prediction_confusion_matrix.png')
    plt.show()


def get_path_df(path: str) -> pd.DataFrame:
    data_dict = {'data_path': [], 'label': []}
    for folder_name in os.listdir(path):
        for file_name in os.listdir(f'{path}/{folder_name}'):
            data_dict['data_path'].append(f'{path}/{folder_name}/{file_name}')
            data_dict['label'].append(folder_name)
    return pd.DataFrame(data_dict)


if __name__ == "__main__":
    main()
