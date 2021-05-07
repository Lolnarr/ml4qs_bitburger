import datetime

import serial
import time

arduino = serial.Serial(port='COM3', baudrate=115200)
f = open("output.csv", "w")
f.write('time,accX,accY,accZ,gyrX,gyrY,gyrZ\n')
f.close()
duration = int(input("Zeitspanne: ")) * 1000
arduino.readline() # Dump old data
starttime = arduino.readline().decode('utf-8').replace('\r\n', '').split(',')[0]
print("Starttime: " + str(starttime))


def currenttime(val):
    if type(val) is str:
        return int(val.split(',')[0]) - int(starttime)
    elif type(val) is int:
        return val - int(starttime)


f = open("output.csv", "a")
value = starttime
counter = 0

while currenttime(value) < int(duration):
    counter += 1
    value = arduino.readline().decode('utf-8').replace('\r\n', '')
#   value = value.replace('\r\n', '')
    if value != "":
        if counter % 100 == 0:
            print("Data/s: " + str(int(1000/(int(value.split(',')[0])/counter))))
        f.write(value + "\n")
#        cumsum = 0
#        for x in value.split(','):
#            cumsum += int(x)
#        f.write("," + str(cumsum) + "\n")

#    time.sleep(0.05)

f.close()
