import os
import serial
import keyboard #Using module keyboard
import tty
import time
import threading
import tkinter as tk

baud  = 115200
fname = 'accel_data.txt'
fmode = 'ab'
reps  = 100

if os.path.exists('/dev/ttyACM0'):
    addr  = '/dev/ttyACM0'
else:
    addr  = '/dev/ttyACM1'

print('using addr', addr)


def onKeyPress(event):
    text.insert('end', 'You pressed %s\n' % (event.char, ))
    print('Key pressed!')
    outf.write(x)
    outf.flush()
    print('done writing')
    root.after(2000, task)


def task():
    outf = open(fname,fmode)
    with serial.Serial(addr,baud) as port:
        x = port.readline()
        print(x)

root = tk.Tk()
root.geometry('300x200')
text = tk.Text(root, background='black', foreground='white', font=('Comic Sans MS', 12))
text.pack()
root.bind('<KeyPress>', onKeyPress)
root.after(2000, task)
root.mainloop()

            # outf.write(x)
            # outf.flush()
    # outf.close()

# #orig_settings = termios.tcgetattr(sys.stdin)
# #tty.setraw(sys.stdin)
# def do_something():
    # with serial.Serial(addr,baud) as port:
        # for i in range(reps):
            # x = port.readline()
            # print(x)
            # # outf.write(x)
            # # outf.flush()
    # # outf.close()


# def _write():
    # print('Writing...')
    # ser = serial.Serial(addr,baud)
    # x = ser.readline()
    # print(x)

    # outf = open(fname,fmode)
    # outf.write(x)
    # outf.flush()
    # print('done writing')

    # # cannot exit with
    # #root.quit()
    # #root.destroy()



# root = Tk.Tk()
# QuitButton = Tk.Button(master=root, text='Write', command=_write) #the quit button
# QuitButton.pack(side=Tk.BOTTOM)

# thread = threading.Thread(target=do_something, args=())
# e = threading.Event()
# thread.start()
# root.mainloop()

# https://stackoverflow.com/questions/11758555/python-do-something-until-keypress-or-timeout

