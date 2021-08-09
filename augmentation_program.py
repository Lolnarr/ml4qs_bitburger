import math
import os
import numpy as np
import pandas as pd
import shutil
from matplotlib import pyplot as plt

TRAIN_PATH = 'normalized_data/training'
TEST_PATH = 'normalized_data/test'
VAL_PATH = 'normalized_data/validation'


def add_noise(df: pd.DataFrame):
    mu, sigma = 0, 400
    noise = np.random.normal(mu, sigma, [100, 6])
    fusion = df + noise
    return fusion


def stretch_data(df: pd.DataFrame):
    mu, sigma = 1, 0.2
    stretch = np.random.normal(mu, sigma, [100, 6])
    fusion = df * stretch
    return fusion
    #accX, accY, accZ, gyrX, gyrY, gyrZ = np.random.randn(6) * 0.3 + 0.7
    #df_new = df * np.array([accX, accY, accZ, gyrX, gyrY, gyrZ])
    # df['gyrX'] = df_new['gyrX']
    # df['gyrY'] = df_new['gyrY']
    # df['gyrZ'] = df_new['gyrZ']
    #return df_new


def start_augmentation(path: str, noise=True, stretch=True):
    for foldername in os.listdir(path):
        if os.path.basename(path) == 'training':
            datafolder = 'training'
        elif os.path.basename(path) == 'test':
            datafolder = 'test'
        elif os.path.basename(path) == 'validation':
            datafolder = 'validation'
        if not os.path.exists(f'augmented_data/{datafolder}/{foldername}'):
            os.makedirs(f'augmented_data/{datafolder}/{foldername}')
        for filename in os.listdir(f'{path}/{foldername}'):
            shutil.copy(f'{path}/{foldername}/{filename}', f'augmented_data/{datafolder}/{foldername}')
            letter = filename.split('.')[0]
            count = 1
            while count <= 5:
                df = pd.read_csv(f'{path}/{foldername}/{filename}')
                if stretch:
                    df = stretch_data(df)
                if noise:
                    df = add_noise(df)
                df.to_csv(f'augmented_data/{datafolder}/{foldername}/{letter}_augmented_{count}.csv', index=False)
                count += 1


def main():
    start_augmentation(TRAIN_PATH)
    start_augmentation(TEST_PATH)
    start_augmentation(VAL_PATH)


if __name__ == '__main__':
    main()
