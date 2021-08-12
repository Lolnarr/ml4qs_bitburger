import time

import pandas as pd
import matplotlib.pyplot as plt


def main():
    data = pd.read_csv(f'normalized_data/training/A/A_10.csv', sep=',', decimal='.')

    fig, axs = plt.subplots(2, sharex=True, sharey=False)
    axs[0].plot(data.index, data['accX'], label='X-axis')
    axs[0].plot(data.index, data['accY'], label='Y-axis')
    axs[0].plot(data.index, data['accZ'], label='Z-axis')
    axs[0].set(title='Accelerometer data', xlim=(0, data.index.max()))
    axs[1].plot(data.index, data['gyrX'], label='X-axis')
    axs[1].plot(data.index, data['gyrY'], label='Y-axis')
    axs[1].plot(data.index, data['gyrZ'], label='Z-axis')
    axs[1].set(title='Gyroscope data', xlim=(0, data.index.max()))
    plt.legend()
    plt.show()
    time.sleep(1)


if __name__ == "__main__":
    main()
