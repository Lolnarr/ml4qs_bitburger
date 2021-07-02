import math
import os
import numpy as np
import pandas as pd

TRAIN_PATH = 'normalized_data/training'
TEST_PATH = 'normalized_data/test'
VAL_PATH = 'normalized_data/validation'


def add_gyronoise(path: str):
    mu, sigma = 0, 1.0
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
            while count <= 1:
                noise = np.random.normal(mu, sigma, [500, 3])
                df = pd.read_csv(f'{path}/{foldername}/{filename}')
                fusion = df[['gyrX', 'gyrY', 'gyrZ']] + noise
                df['gyrX'] = fusion['gyrX']
                df['gyrY'] = fusion['gyrY']
                df['gyrZ'] = fusion['gyrZ']
                df.to_csv(f'augmented_data/{datafolder}/{foldername}/{letter}_noise_{count}.csv', index=False)
                count += 1


def add_noise(path: str):
    mu, sigma = 0, 1.0
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
            while count <= 1:
                noise = np.random.normal(mu, sigma, [500, 6])
                df = pd.read_csv(f'{path}/{foldername}/{filename}')
                fusion = df + noise
                fusion.to_csv(f'augmented_data/{datafolder}/{foldername}/{letter}_noise_{count}.csv', index=False)
                count += 1


def stretch_data(path: str):
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
                accX, accY, accZ, gyrX, gyrY, gyrZ = np.random.randn(6) * 0.3 + 1
                df = pd.read_csv(f'{path}/{foldername}/{filename}')
                df = df * np.array([accX, accY, accZ, gyrX, gyrY, gyrZ])
                df.to_csv(f'augmented_data/{datafolder}/{foldername}/{letter}_stretch_{count}.csv', index=False)
                count += 1


def stretch_gyrodata(path: str):
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
                gyrX, gyrY, gyrZ = np.random.randn(3) * 0.3 + 1
                df = pd.read_csv(f'{path}/{foldername}/{filename}')
                df_new = df[['gyrX', 'gyrY', 'gyrZ']] * np.array([gyrX, gyrY, gyrZ])
                df['gyrX'] = df_new['gyrX']
                df['gyrY'] = df_new['gyrY']
                df['gyrZ'] = df_new['gyrZ']
                df.to_csv(f'augmented_data/{datafolder}/{foldername}/{letter}_stretch_{count}.csv', index=False)
                count += 1


def rotate_data(path: str):
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
            while count <= 1:
                df = pd.read_csv(f'{path}/{foldername}/{filename}')
                rotation_axis = np.array([0, 0, 1])
                theta = np.random.randn() * 5
                q = np.hstack([np.array(math.cos(np.radians(theta / 2.0))),
                               rotation_axis * math.sin(np.radians(theta / 2.0))])
                a, b, c, d = q
                df_new = np.matmul(
                    np.array([[a ** 2 + b ** 2 - c ** 2 - d ** 2, 2 * b * c - 2 * a * d, 2 * b * d + 2 * a * c],
                              [2 * b * c + 2 * a * d, a ** 2 - b ** 2 + c ** 2 - d ** 2, 2 * c * d - 2 * a * b],
                              [2 * b * d - 2 * a * c, 2 * c * d + 2 * a * b, a ** 2 - b ** 2 - c ** 2 + d ** 2]]),
                    df[['gyrX', 'gyrY', 'gyrZ']].T).T
                df_new.columns = ['gyrX', 'gyrY', 'gyrZ']
                df['gyrX'] = df_new['gyrX']
                df['gyrY'] = df_new['gyrY']
                df['gyrZ'] = df_new['gyrZ']
                df.to_csv(f'augmented_data/{datafolder}/{foldername}/{letter}_rotate_{count}.csv', index=False)
                count += 1


# def quaternion_to_rotation_matrix(q):
#     a, b, c, d = q
#     return np.array([[a ** 2 + b ** 2 - c ** 2 - d ** 2, 2 * b * c - 2 * a * d, 2 * b * d + 2 * a * c],
#                      [2 * b * c + 2 * a * d, a ** 2 - b ** 2 + c ** 2 - d ** 2, 2 * c * d - 2 * a * b],
#                      [2 * b * d - 2 * a * c, 2 * c * d + 2 * a * b, a ** 2 - b ** 2 - c ** 2 + d ** 2]])


def main():
    add_gyronoise(TRAIN_PATH)
    add_gyronoise(TEST_PATH)
    add_gyronoise(VAL_PATH)


if __name__ == '__main__':
    main()
