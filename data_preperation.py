import os
import pandas as pd

DATA_PATH = 'recorded_data/LennartB'


def split_data(df: pd.DataFrame, letter: str):
    if not os.path.exists(f'split_data/{letter}'):
        os.mkdir(f'split_data/{letter}')
    count = get_count(letter) + 1
    prev_time = 0
    for index, row in df.iterrows():
        if row['time'] < prev_time:
            df_new = df[df.index < index]
            df.drop(df[df.index < index].index, inplace=True)
            df_new.to_csv(f'split_data/{letter}/{letter}_{count}.csv')
            count += 1
        prev_time = row['time']
    df.to_csv(f'split_data/{letter}/{letter}_{count}.csv')


def get_count(letter: str) -> int:
    path = f'split_data/{letter}'
    count = 1
    for filename in os.listdir(path):
        i = int(filename.split('.')[0].split('_')[1])
        count = i if i > count else count
    return count


def main():
    for filename in os.listdir(DATA_PATH):
        letter = filename.split('.')[0]
        df = pd.read_csv(DATA_PATH + '/' + filename)
        split_data(df, letter)


if __name__ == '__main__':
    main()
