import math
import os
import numpy as np
import pandas as pd

TRAIN_PATH = 'normalized_data/training'
TEST_PATH = 'normalized_data/test'
VAL_PATH = 'normalized_data/validation'


def add_gyronoise(df: pd.DataFrame):
    mu, sigma = 0, 1.0
    noise = np.random.normal(mu, sigma, [500, 3])
    fusion = df[['gyrX', 'gyrY', 'gyrZ']] + noise
    df['gyrX'] = fusion['gyrX']
    df['gyrY'] = fusion['gyrY']
    df['gyrZ'] = fusion['gyrZ']
    return df


def add_noise(df: pd.DataFrame):
    mu, sigma = 0, 1.0
    noise = np.random.normal(mu, sigma, [500, 9])
    fusion = df + noise #df[['roll', 'pitch', 'yaw']] + noise
    return fusion


def stretch_data(df: pd.DataFrame):
    r, p, y = np.random.randn(3) * 0.3 + 1
    df = df[['roll', 'pitch', 'yaw']] * np.array([r, p, y])
    return df


def stretch_gyrodata(df: pd.DataFrame):
    accX, accY, accZ, gyrX, gyrY, gyrZ, r, p, y = np.random.randn(9) * 0.3 + 1
    df_new = df * np.array([accX, accY, accZ, gyrX, gyrY, gyrZ, r, p, y])
    # df['gyrX'] = df_new['gyrX']
    # df['gyrY'] = df_new['gyrY']
    # df['gyrZ'] = df_new['gyrZ']
    return df_new


def rotate_data(df: pd.DataFrame):
    rotation_axis = np.array([0, 0, 1])
    theta = np.random.randn() * 5
    q = np.hstack([np.array(math.cos(np.radians(theta / 2.0))),
                   rotation_axis * math.sin(np.radians(theta / 2.0))])
    a, b, c, d = q
    df_new = np.matmul(np.array([[a ** 2 + b ** 2 - c ** 2 - d ** 2, 2 * b * c - 2 * a * d, 2 * b * d + 2 * a * c],
                                 [2 * b * c + 2 * a * d, a ** 2 - b ** 2 + c ** 2 - d ** 2, 2 * c * d - 2 * a * b],
                                 [2 * b * d - 2 * a * c, 2 * c * d + 2 * a * b, a ** 2 - b ** 2 - c ** 2 + d ** 2]]),
                       df[['accX', 'accY', 'accZ', 'gyrX', 'gyrY', 'gyrZ', 'roll', 'pitch', 'yaw']].T).T
    #df_new.columns = ['roll', 'pitch', 'yaw']
    # df['gyrX'] = df_new['gyrX']
    # df['gyrY'] = df_new['gyrY']
    # df['gyrZ'] = df_new['gyrZ']
    return df_new


def start_augmentation(path: str, rotate=False, gyronoise=False, noise=True, gyrostretch=True, stretch=False):
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
            letter = filename.split('.')[0]
            count = 1
            while count <= 2:
                df = pd.read_csv(f'{path}/{foldername}/{filename}')
                if rotate:
                    df = rotate_data(df)
                if gyrostretch:
                    df = stretch_gyrodata(df)
                if stretch:
                    df = stretch_data(df)
                if gyronoise:
                    df = add_gyronoise(df)
                if noise:
                    df = add_noise(df)
                df.to_csv(f'augmented_data/{datafolder}/{foldername}/{letter}_augmented_{count}.csv', index=False)
                count += 1


def main():
    #start_augmentation(TRAIN_PATH)
    start_augmentation(TEST_PATH)
    #start_augmentation(VAL_PATH)


if __name__ == '__main__':
    main()
