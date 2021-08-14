import os
import pandas as pd
from ahrs.filters import Madgwick

DATA_PATH = 'predict_data'


def add_orientation_data(path: str):
    #for partition in os.listdir(f'{path}'):
    for folder_name in os.listdir(path):
        if not os.path.exists(f'ahrs_data/{folder_name}'):
            os.makedirs(f'ahrs_data/{folder_name}')
        for file_name in os.listdir(f'{path}/{folder_name}'):
            df = pd.read_csv(f'{path}/{folder_name}/{file_name}')
            acc_data = df[['accX', 'accY', 'accZ']]
            gyro_data = df[['gyrX', 'gyrY', 'gyrZ']]
            orientation = Madgwick(gyr=gyro_data.values, acc=acc_data.values)
            quaternions = orientation.Q
            df[['q1', 'q2', 'q3', 'q4']] = quaternions
            df.to_csv(f'ahrs_data/{folder_name}/{file_name}')


def main():
    add_orientation_data(DATA_PATH)


if __name__ == '__main__':
    main()
