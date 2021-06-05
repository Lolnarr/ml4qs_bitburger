import time

import pandas as pd
import matplotlib.pyplot as plt

def main(letter):
    # Read in data using pandas
    for i in range(21):
        # file = input("Dateiname (xyz.csv): ")
        data = pd.read_csv(f'split_data/{letter}/{letter}_{i+1}.csv', sep=',', decimal='.')

        fig, axs = plt.subplots(2, sharex=True, sharey=False)
        axs[0].plot(data['time'], data['accX'], label='X-axis')
        axs[0].plot(data['time'], data['accY'], label='Y-axis')
        axs[0].plot(data['time'], data['accZ'], label='Z-axis')
        axs[0].set(title='Accelerometer data', xlim=(0, data['time'].max()))
        axs[1].plot(data['time'], data['gyrX'], label='X-axis')
        axs[1].plot(data['time'], data['gyrY'], label='Y-axis')
        axs[1].plot(data['time'], data['gyrZ'], label='Z-axis')
        axs[1].set(title='Gyroscope data', xlim=(0, data['time'].max()))
        plt.legend()
    #    plt.savefig(f'recorded_data/{file}.png')
        plt.show()
        time.sleep(1)


# Plotting the axes in individual plots
"""
fig, axs = plt.subplots(6, sharex=True, sharey=True)
for index, axis in enumerate(['accX', 'accY', 'accZ', 'gyrX', 'gyrY', 'gyrZ']):
    axs[index].plot(data['time'], data[axis])
    axs[index].set(title=f'{axis}-Axis', xlim=(0, data['time'].max()))
plt.show()
"""

if __name__ == "__main__":
    while True:
        main(input("Buchstabe? ").upper())
