'''
9 April 2018
This file caculates the torques across all 492 datapoints and creates a CSV of format
1 - posx posy posz - thetax thetay thetaz -  etc.
|
v
nth datapoint
'''
import math
import shelve

import pandas as pd
import numpy as np

path = "~/Documents/projects_Spring2018/howe299r/Experiments/03April2018/WIP/"
#IMUDats = [ '%02dIMU.txt'% x for x in pos ]
IMUCols = ['timeSysCal', 'XYZ','X', 'Y', 'Z']

#===============================================
#### DECLARE CONSTANTS ####
#===============================================
BigTheta = np.zeros((1,3))
BigTorque = np.zeros((1,3))
BigForce = np.zeros((1,3))
BigPosition = np.zeros((1,3))
BigPosIdx = []


listpos = range(15)
#listpos = [15]
xs = [4.6, 4.1, 3.5, 3.1, 2.6] #pos 1, x coord = 4.6 cm
ys = [0.4, 0.1, -0.2]
posX = np.repeat(xs,3)
posY = np.tile(ys, 5)
posZ = np.array([0]*15)

#===============================================
#### ACCUMULATE DATA ACROSS 15 POSITIONS  ####
#===============================================
for i in listpos:
    print('position number: ', i+1)
    #### READ DATA ####
    fname =  '%02dIMU.txt'%(i+1)
    imuDat = pd.read_csv(path+fname, header=None,sep=';', 
                        names = IMUCols, skip_blank_lines=True, usecols=[0,1,2,3,4]) 
    imuDat = imuDat.drop(['timeSysCal', 'XYZ'], 1)
    bkgd = imuDat.iloc[0::2]  
    signal = imuDat.iloc[1::2]
    zer = bkgd.as_matrix()
    sig = signal.as_matrix()

    #### DECLARE CONSTANTS ####
    pos = np.array([posX[i], posY[i], posZ[i]])
    thetas = sig-zer

    # we have some issues with overflow of 3rd col; detect and compensate
    overflow_idx = np.where(thetas[:,2] < -300)
    thetas[overflow_idx, :] += np.array([0,0,2*179])

    #IMU.z is roll IRL. But in our coordinate system, torque_z = yaw
    thetas = np.fliplr(thetas)

    n = thetas.shape[0] # num datapoints
    #print('num datapoints: ',n)

    #### CALCULATE TORQUE ####
    forcesZ = [[20*f]*3 for f in range(1, int(n/3)+1)] #start at 20g
    forcesZ = np.array(forcesZ).flatten()
    forces = np.column_stack((np.zeros((n,2)),forcesZ)) #n.3

    torques = np.cross(pos.reshape(-1,1).T, forces) #n.3
    #print('torques\n', torques)
    BigTheta = np.vstack((BigTheta, thetas))
    BigTorque = np.vstack((BigTorque, torques))
    positions = np.tile(pos, n).reshape(n, -1)
    BigPosition = np.vstack((BigPosition, positions))
    BigForce = np.vstack((BigForce, forces))
    BigPosIdx.extend([i]*n)

BigTheta = BigTheta[1:,:] #remove first row of zeros, from init
BigTorque = BigTorque[1:,:]
BigPosition = BigPosition[1:,:] #remove first row of zeros, from init
BigForce = BigForce[1:,:]
BigPosIdx = np.array(BigPosIdx)
print(BigPosIdx)

print('number of datapoints', BigTheta.shape)

with shelve.open('calculated_data', 'c') as shelf:
    shelf['BigTheta'] = BigTheta
    shelf['BigTorque'] = BigTorque
    shelf['listpos'] = listpos 
    shelf['BigForce'] = BigForce
    shelf['BigPosition'] = BigPosition
    shelf['BigPosIdx'] = BigPosIdx 
    # shelf['forces'] = forces 
    # shelf['posXYZ'] = posXYZ 


# print(BigPosition.shape)
# print(BigForce.shape)
# print(BigTheta.shape)
# print(BigTorque.shape)

