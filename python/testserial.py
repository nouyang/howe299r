import os
import serial
import keyboard #Using module keyboard


baud  = 115200
fname = 'accel_data.txt'
fmode = 'ab'
reps  = 1000

if os.path.exists('/dev/ttyACM0'):
    addr  = '/dev/ttyACM0'
else:
    addr  = '/dev/ttyACM1'
    
print('using addr', addr)


with serial.Serial(addr,baud) as port, open(fname,fmode) as outf:
    for i in range(reps):
        x = port.readline()
        print(x)
        outf.write(x)
        outf.flush()

outf.close()
