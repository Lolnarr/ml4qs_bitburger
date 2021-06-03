import pandas as pd
import serial
from pygame import mixer
import time
from pathlib import Path

mixer.init()
mixer.music.load("beep.mp3")

df_lettercount = pd.read_csv('letter_count.csv', sep=',', decimal='.')
print(df_lettercount)

duration = int(input("Zeitspanne: ")) * 1000

arduino = serial.Serial(port='/dev/cu.usbmodem142301', baudrate=115200)  # Establish serial connection
for i in range(10):  # Dump old data
    arduino.readline()
    if i == 9:
        start_value = arduino.readline().decode('utf-8').replace('\r\n', '')


def currenttime(val):
    if type(val) is str:
        return int(val.split(',')[0]) - int(starttime)
    elif type(val) is int:
        return val - int(starttime)


def datamining():
    global start_value
    value = start_value
    counter = 0
    df = pd.DataFrame(columns=['time', 'accX', 'accY', 'accZ', 'gyrX', 'gyrY', 'gyrZ'])
    # temp_df = pd.DataFrame(columns=['time', 'accX', 'accY', 'accZ', 'gyrX', 'gyrY', 'gyrZ'])
    while currenttime(value) < int(duration):
        counter += 1
        value = arduino.readline().decode('utf-8').replace('\r\n', '')
        print(value)
        if value != "":
            if counter % 100 == 0:
                print("Data/s: " + str(int(1000/(int(value.split(',')[0])/counter))))
            split_arr = value.split(',')
            df = df.append({'time': int(split_arr[0]) - int(starttime),
                            'accX': split_arr[1],
                            'accY': split_arr[2],
                            'accZ': split_arr[3],
                            'gyrX': split_arr[4],
                            'gyrY': split_arr[5],
                            'gyrZ': split_arr[6]},
                            ignore_index=True)
    return df

def dump_data():
    for i in range(2000):
        arduino.readline()


input_letter = ''
prev_letter = ''
file_num = 0

while input_letter != "exit".upper():
    data_arr = pd.DataFrame(columns=['time', 'accX', 'accY', 'accZ', 'gyrX', 'gyrY', 'gyrZ'])
    input_letter = str(input("Buchstabe OR . OR exit: ")).upper()
    if input_letter != "" and not input_letter == ".":
        prev_letter = input_letter
    else:
        if input_letter == ".":
            df_lettercount.loc[df_lettercount['letter'] == prev_letter, "count"] = file_num
        input_letter = prev_letter
    if input_letter == "exit".upper():
        continue

    print("Letter: " + input_letter)
    file_num = df_lettercount.loc[df_lettercount['letter'] == input_letter, 'count'].values[0]
    print("Count: " + str(file_num+1))
    #Path(f"/recorded_data/{input_letter.capitalize()}").mkdir(parents=True, exist_ok=True)
    dump_data()
    mixer.music.play()
    starttime = arduino.readline().decode('utf-8').replace('\r\n', '').split(',')[0]
    print("Starttime: " + str(starttime))
    data_arr = datamining()
    mixer.music.play()
    print(data_arr)
    data_arr.to_csv(f"recorded_data/{input_letter}{file_num}.csv", index=False)
    # df_lettercount = df_lettercount.set
    df_lettercount.loc[df_lettercount['letter'] == input_letter, "count"] = file_num + 1
    df_lettercount.to_csv("letter_count.csv", index=False)
    # f = open(f"recorded_data/{input_letter}{file_num}.csv", "w+")
    # f.write('time,accX,accY,accZ,gyrX,gyrY,gyrZ\n')
    # f.close()
