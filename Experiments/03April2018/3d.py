"""
Created on Thu Apr 9 
@author: nrw
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import plotly.plotly as py
import plotly.offline as po
import plotly.graph_objs as go
from sklearn import linear_model
from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error, r2_score

path = "~/Documents/projects_Spring2018/howe299r/Experiments/03April2018/WIP/"
#IMUDats = [ '%02dIMU.txt'% x for x in pos ]
IMUCols = ['timeSysCal', 'XYZ','X', 'Y', 'Z']

BigTheta = np.zeros((1,3))
BigTorque = np.zeros((1,3))


#===============================================
#### DECLARE CONSTANTS ####
#===============================================
listpos = range(15)
listpos = [0]
#listpos = range(1,3)
xs = [4.6, 4.1, 3.5, 3.1, 2.6]#pos 1 x = 4.6 cm
#ys = [0.4, 0.1, -0.2]
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
 #IMU.z is roll IRL. But in our coordinate system, torque_z = yaw
    thetas = np.fliplr(thetas)
    #thetaX, thetaY, thetaZ = thetas[:,0], thetas[:,1], thetas[:,2]
    n = thetas.shape[0] # num datapoints
    print('num datapoints: ',n)

    #### CALCULATE TORQUE ####
    forcesZ = [20*f for f in range(1, n+1)] #start at 20g
    forces = np.column_stack((np.zeros((n,2)),forcesZ)) #n.3
    #print(forces)

    torques = np.cross(pos.reshape(-1,1).T, forces) #n.3
    print('3D torques:\n', torques)
    print('3D deflections: \n', thetas)
    #torquesX, torquesY, torquesZ = torques[:,0] , torques[:,1], torques[:,2]
    BigTheta = np.vstack((BigTheta, thetas))
    BigTorque = np.vstack((BigTorque, torques))

BigTheta = BigTheta[1:,:] #remove first row of zeros, from init
BigTorque = BigTorque[1:,:]

print('number of datapoints', BigTheta.shape)

#### CALCULATE K ####
matK = np.linalg.lstsq(BigTorque, BigTheta, rcond=None)[0]
print(matK.shape)
print(matK)

'''
#===============================================
#### PLOT ####
#===============================================

## Note: For the IMU, orientation.Y is pitch; X is roll; Z is yaw

torquesY = BigTorque
thetaY = BigTheta
trace0 = go.Scatter( x = thetaY , y = torquesY, mode = 'markers',
    name = 'degrees (by IMU)' )

myX = thetaY.reshape(-1,1)
myy = torquesY

#regr= Ridge(fit_intercept=True, alpha=1.0, random_state=0, normalize=True)
regr = linear_model.LinearRegression()
regr.fit(myX, myy)
coef_ridge = regr.coef_
gridx = np.linspace(myX.min(), myX.max(), 20)
coef_ = regr.coef_ * gridx + regr.intercept_
print(regr.coef_)
print(regr.intercept_)
yPred= regr.predict(myX) 
#plt.plot(gridx, coef_, 'g-', label="ridge regression")

trace2 = go.Scatter( x= gridx, y = coef_,
    name = 'linear fit (w/ridge penalty)' )

data = [trace0, trace2]
    
layout = go.Layout(
    title='Pitch: degrees of deflection vs torque',
    yaxis=dict(title='torque (in grams)'),
    xaxis=dict(title='degrees'),
    legend=dict(x=.1, y=-.5) )

fig = go.Figure(data=data, layout=layout)

print('Coefficients: \n', regr.coef_)
print("Mean squared error: %.2f" % mean_squared_error(thetaY, yPred))
#print('Variance score (ideal 1): %.2f' % r2_score(thetaY))
po.plot(fig)

'''
