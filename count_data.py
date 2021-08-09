import os
import pandas as pd

PATH = 'split_data'


def main():
    print(calculate_frequency(PATH, count_data(PATH), 3))


def count_data(path: str):
    count = 0
    for foldername in os.listdir(path):
        for filename in os.listdir(f'{path}/{foldername}'):
            count = count + 1
    return count


def calculate_frequency(path: str, letters: int, seconds: int):
    length = 0
    for foldername in os.listdir(path):
        for filename in os.listdir(f'{path}/{foldername}'):
            df = pd.read_csv(f'{path}/{foldername}/{filename}')
            length = length + len(df)
    hz = length / letters / seconds
    return hz


if __name__ == '__main__':
    main()
