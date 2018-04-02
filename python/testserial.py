import os
import serial
import keyboard #Using module keyboard
import tty
import time
import threading
import tkinter as Tk

baud  = 115200
fname = 'accel_data.txt'
fmode = 'ab'
reps  = 100

if os.path.exists('/dev/ttyACM0'):
    addr  = '/dev/ttyACM0'
else:
    addr  = '/dev/ttyACM1'

print('using addr', addr)


#orig_settings = termios.tcgetattr(sys.stdin)
#tty.setraw(sys.stdin)
def do_something():
    with serial.Serial(addr,baud) as port, open(fname,fmode) as outf:
        for i in range(reps):
            x = port.readline()
            print(x)
    outf.close()


def _quit():
    print('Exiting...')
    e.set()
    thread.join() #wait for the thread to finish
    root.quit()
    root.destroy()



root = Tk.Tk()
QuitButton = Tk.Button(master=root, text='Quit', command=_quit) #the quit button
QuitButton.pack(side=Tk.BOTTOM)

thread = threading.Thread(target=do_something, args=())
e = threading.Event()
thread.start()
root.mainloop()
