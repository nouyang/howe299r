import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dfgui

path = "~/Documents/projects_Spring2018/howe299r/Experiments/03April2018/WIP/"
pos = range(1,16)
#CVDats = [ '%02dopenCV.txt'% x for x in pos ]
IMUDats = [ '%02dIMU.txt'% x for x in pos ]
# CVCols = ['timestamp', 'distance', 'x' , 'y', 'z', 'yaw', 'pitch', 'roll']
IMUCols = ['timeSysCal', 'XYZ', 'X', 'Y', 'Z']

# cvdat = pd.read_csv(path+'01openCV.txt', sep=';',
                    # names = CVCols) #TODO: this is somehow skipping a col, so the last is NaN

# cvdat = cvdat.drop('timestamp', 1)
# cvdat = cvdat.drop('distance', 1)
# openCVDat = names = IMUCols) #TODO: this is somehow skipping a col, so the last is NaN
imuDat = pd.read_csv(path+'01IMU.txt', sep=';',
                    names = IMUCols) #TODO: this is somehow skipping a col, so the last is NaN

print(imuDat)








# TODO
# shave off empty lines
# remove 1st column on CV files (timestamp)
# remove 1st 2 columns on IMU files (timestamp)
# subtract remaining even lines from prev odd lines = delta positions
# annotate with x positions
