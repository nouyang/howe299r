"""
@ nouyang
04 May 2018

Python side of arduino-python real-time plotting.
Axes are time (milliseconds) and pressure in Pa (from BMP280 via SPI)
See analog-plot.ino for matching python code
Assumes serial port is either ACM1 or ACM0 (as on Ubuntu using Uno)

Plotting: Uses code from electronut.in Mahesh Venkitachalam
"""

import sys, serial, argparse
import numpy as np
from time import sleep
from collections import deque
import os

import matplotlib.pyplot as plt 
import matplotlib.animation as animation

      
# plot class
class AnalogPlot:
    # constr
    def __init__(self, strPort, maxLen):
        # open serial port
        self.ser = serial.Serial(strPort, 115200)

        self.ax = deque([0.0]*maxLen)
        self.ay = deque([0.0]*maxLen)
        self.maxLen = maxLen

    # add to buffer
    def addToBuf(self, buf, val):
        if len(buf) < self.maxLen:
            buf.append(val)
        else:
            buf.pop()
            buf.appendleft(val)

    # add data
    def add(self, data):
        assert(len(data) == 2)
        self.addToBuf(self.ax, data[0])
        self.addToBuf(self.ay, data[1])

    # update plot
    def update(self, frameNum, a0, a1):
        try:
            line = self.ser.readline()
            data = [float(val) for val in line.split()]
            # print data
            if(len(data) == 2):
                self.add(data)
                print('data', data)
                a0.set_data(range(self.maxLen), self.ax)
                a1.set_data(range(self.maxLen), self.ay)
        except KeyboardInterrupt:
            print('exiting')
        
        return a0

    # clean up
    def close(self):
        # close serial
        self.ser.flush()
        self.ser.close()    

###----------  MAIN  -----------

def main():
    baud  = 115200
    # strtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if os.path.exists('/dev/ttyACM0'):
      addr  = '/dev/ttyACM0'
    else:
      addr  = '/dev/ttyACM1'

    print('using addr', addr)

    ser = serial.Serial(addr,baud)

    # # create parser
    # parser = argparse.ArgumentParser(description="LDR serial")
    # # add expected arguments
    # parser.add_argument('--port', dest='port', required=True)

    # # parse args
    # args = parser.parse_args()
    
    # #strPort = '/dev/tty.usbserial-A7006Yqh'
    # strPort = args.port

    print('reading from serial port %s...' % addr)

    # plot parameters
    analogPlot = AnalogPlot(addr, 100)

    print('plotting data...')

    # calculate y axis limits
    line = ser.readline()
    somedata = [float(val) for val in line.split()]
    maxy = max(somedata)
    miny = min(somedata)
    print(somedata)
    print(len(somedata))
    print(miny, maxy)
    ymin = miny*0.85
    ymax  = maxy*1.15
    print("setting ylims to", ymin, ymax)

    # set up animation
    fig = plt.figure() 
    ax = plt.axes(xlim=(0, 100), ylim=(ymin, ymax), 
                  ylabel="Pressure (Pa)", xlabel="Time (# of datapoints)",
                  title="BMP280 Pressure Sensors | 06 May 2018")
    a0, = ax.plot([], [])
    a1, = ax.plot([], [])
   
    anim = animation.FuncAnimation(fig, analogPlot.update, 
                                   fargs=(a0, a1), 
                                   interval=50)

    # show plot
    plt.show()
    
    # clean up
    analogPlot.close()

    print('exiting.')
        

    #https://matplotlib.org/api/animation_api.html
    # http://scipy-cookbook.readthedocs.io/items/Matplotlib_Animations.html
    # https://matplotlib.org/examples/animation/index.html
    # https://stackoverflow.com/questions/13181118/using-matplotlib-or-pyqtgraph-to-graph-real-time-data
    # https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/

# call main
if __name__ == '__main__':
    main()
