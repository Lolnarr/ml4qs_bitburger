import os
import pandas as pd
import numpy as np


DATA_PATH = 'predict_data'
LETTER = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def split_data(df: pd.DataFrame, letter: str):
    if not os.path.exists(f'DATA/predict_data/{letter}'):
        os.makedirs(f'DATA/predict_data/{letter}')
    count = get_count(letter) + 1
    prev_time = 0
    for index, row in df.iterrows():
        if row['time'] < prev_time:
            df_new = df[df.index < index]
            df.drop(df[df.index < index].index, inplace=True)
            df_new.to_csv(f'DATA/predict_data/{letter}/{letter}_{count}.csv')
            count += 1
        prev_time = row['time']
    df.to_csv(f'DATA/predict_data/{letter}/{letter}_{count}.csv')


def split_git_data(path: str):
    for folder_name in os.listdir(path):
        print(folder_name)
        for file_name in os.listdir(f'{path}/{folder_name}'):
            if file_name.split('.')[0] in LETTER and len(file_name) == 5:
                label = file_name.split('.')[0]
                print(label)
                if not os.path.exists(f'git_data/split_data/{label}'):
                    os.mkdir(f'git_data/split_data/{label}')
                df = pd.read_csv(f'{path}/{folder_name}/{file_name}',
                                 names=['id', 'time delta', 'yaw', 'pitch', 'roll',
                                        'ax1', 'ay1', 'az1', 'ax2', 'ay2', 'az2', 'ax3', 'ay3', 'az3', 'c'])
                count = get_count(f'git_data/split_data/{label}')
                prev_id = df['id'].iloc[1]
                for index, row in df.iterrows():
                    if row['id'] > prev_id:
                        df_new = df[df.index < index]
                        df.drop(df[df.index < index].index, inplace=True)
                        df_new.to_csv(f'git_data/split_data/{label}/{label}_{count}.csv')
                        count += 1
                    prev_id = row['id']


def normalize_data(path: str):
    for folder_name in os.listdir(path):
        i = 0
        for file_name in os.listdir(f'{path}/{folder_name}'):
            num_letters = len(os.listdir(f'{path}/{folder_name}'))
            per_training = round(num_letters * 0.6)  # 0.6
            per_validation = round(num_letters * 0.1)  # 0.1
            per_test = round(num_letters * 0.3)  # 0.3

            df = pd.read_csv(f'{path}/{folder_name}/{file_name}', index_col=0)
            df.drop(columns=['time'], inplace=True)
            # df.drop(columns=['time delta', 'id', 'ax1', 'ay1', 'az1', 'ax2', 'ay2', 'az2', 'ax3', 'ay3', 'az3', 'c'],
            #        inplace=True)
            df = resample_fixed(df, 100)
            if i <= per_training:
                partition = 'training'
            elif i <= per_validation+per_training:
                partition = 'validation'
            if i <= per_test+per_validation+per_training:
                partition = 'test'

            # partition = 'test'
            # partition = np.random.choice(['training', 'validation', 'test'], 1, p=[0.7, 0.1, 0.2])
            if not os.path.exists(f'DATA/normalized_data/predict_data/{partition}/{folder_name}'): #f'bitburger_testdata/{folder_name}'
                os.makedirs(f'DATA/normalized_data/predict_data/{partition}/{folder_name}') #f'bitburger_testdata/{folder_name}
            df.to_csv(f'DATA/normalized_data/predict_data/{partition}/{folder_name}/{file_name}', index=False) #f'bitburger_testdat/{folder_name}/{file_name}
            i += 1


def resample_fixed(df: pd.DataFrame, n_new: int) -> pd.DataFrame:
    n_old, m = df.values.shape
    mat_old = df.values
    mat_new = np.zeros((n_new, m))
    x_old = np.linspace(df.index.min(), df.index.max(), n_old)
    x_new = np.linspace(df.index.min(), df.index.max(), n_new)

    for j in range(m):
        y_old = mat_old[:, j]
        y_new = np.interp(x_new, x_old, y_old)
        mat_new[:, j] = y_new

    return pd.DataFrame(mat_new, columns=df.columns)


def get_count(letter: str) -> int:
    path = f'DATA/predict_data/{letter}'
    count = 0
    for filename in os.listdir(path):
        i = int(filename.split('.')[0].split('_')[1])
        count = i if i > count else count
    return count


def main():
    normalize_data(DATA_PATH)
    '''
    for filename in os.listdir(DATA_PATH):
        letter = filename.split('.')[0]
        df = pd.read_csv(DATA_PATH + '/' + filename)
        split_data(df, letter)
    '''


if __name__ == '__main__':
    main()
