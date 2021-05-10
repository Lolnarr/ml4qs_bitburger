import pandas as pd
import matplotlib.pyplot as plt

# Read in data using pandas
data = pd.read_csv('output.csv', sep=',', decimal='.')
# Rename columns
# data.columns = ['time', 'x', 'y', 'z', 'absolute']

# Plotting all axis in one single plot
"""
plt.plot(data['time'], data['accX'], label='X-axis')
plt.plot(data['time'], data['accY'], label='Y-axis')
plt.plot(data['time'], data['accZ'], label='Z-axis')
plt.legend()
plt.title('Accelerometer data')
plt.xlim(0, data['time'].max())
plt.show()
"""

# Plotting the axes in individual plots
fig, axs = plt.subplots(3, sharex=True, sharey=False)
for index, axis in enumerate(['accX', 'accY', 'accZ']):
    axs[index].plot(data['time'], data[axis])
    axs[index].set(title=f'{axis}-Axis', xlim=(0, data['time'].max()))
plt.show()