import os
import numpy as np
import pandas as pd

TRAIN_PATH = 'normalized_data/training'
TEST_PATH = 'normalized_data/test'
VAL_PATH = 'normalized_data/validation'


def add_noise(path: str):
    mu, sigma = 0, 0.1
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
            while count <= 3:
                noise = np.random.normal(mu, sigma, [500, 7])
                df = pd.read_csv(f'{path}/{foldername}/{filename}')
                fusion = df + noise
                fusion.to_csv(f'augmented_data/{datafolder}/{foldername}/{letter}_noise_{count}.csv')
                count += 1


def main():
    add_noise(TRAIN_PATH)
    add_noise(TEST_PATH)
    add_noise(VAL_PATH)


if __name__ == '__main__':
    main()
