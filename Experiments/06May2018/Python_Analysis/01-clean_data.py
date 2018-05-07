'''
Created on 07 May 2018
@author: nrw

Takes IMU files from final dataset (06 May)
Shelves all the data:
-- position force applied XYZ, theta deflected XYZ, torque XYZ, force applied XYZ
'''
import math
import shelve

import pandas as pd
import numpy as np

path = "~/Documents/projects_Spring2018/howe299r/Experiments/06May2018/Python_Analysis/IMU_data_no_tendon/"
#IMUDats = [ '%02dIMU.txt'% x for x in pos ]
IMUCols = ['timeSysCal', 'XYZ','X', 'Y', 'Z']
ForcesList = [20,50,70,100,150,200]

#===============================================
#### DECLARE CONSTANTS ####
#===============================================
BigTheta = np.zeros((1,3))
BigTorque = np.zeros((1,3))
BigForce = np.zeros((1,3))
BigPosition = np.zeros((1,3))
BigPosIdx = []


listpos = [4,5,6,10,11,12]
#listpos = [15]
xs = [4.6, 4.1, 3.5, 3.1, 2.6] #pos 1, x coord = 4.6 cm
ys = [0.2, -0.1, -0.3]
posX = np.repeat(xs,3)
posY = np.tile(ys, 5)
posZ = np.array([0]*15)

#===============================================
#### ACCUMULATE DATA ACROSS 15 POSITIONS  ####
#===============================================
for i in listpos:
    print('position number: ', i)
    #### READ DATA ####
    fname =  '%02d.txt'%(i)
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
    print('Thetas: ', thetas)

    # we have some issues with overflow of 3rd col; detect and compensate
    overflow_idx = np.where(thetas[:,2] < -300)
    thetas[overflow_idx, :] += np.array([0,0,2*179])

    #IMU.z is roll IRL. But in our coordinate system, torque_z = yaw
    thetas = np.fliplr(thetas)

    n = thetas.shape[0] # num datapoints
    #print('num datapoints: ',n)

    #### CALCULATE TORQUE ####
    # we simplify model as only have force in z direction
    # 36 datapoints / 3 samples pre position / 2 (zero and applied force) datapoints per sample
    numforces = int(imuDat.shape[0]/6)
    possibForces = np.array(ForcesList[0:numforces])
    # print('possib forces', possibForces)
    forcesZ = np.repeat(possibForces, 3) 
    # print('forcesZ', forcesZ))

    # print('numforces', numforces, 'forcesZ shape', forcesZ.shape)
    # print('n', n)
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

with shelve.open('cleaned_data', 'c') as shelf:
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

