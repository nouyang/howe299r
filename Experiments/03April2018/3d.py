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
listpos = [12]
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
    #print(imuDat.describe())
    #print(imuDat.head())
    bkgd = imuDat.iloc[0::2]  
    signal = imuDat.iloc[1::2]
    zer = bkgd.as_matrix()
    sig = signal.as_matrix()

    #### DECLARE CONSTANTS ####
    pos = np.array([posX[i], posY[i], posZ[i]])
    thetas = sig-zer
    # np.delete(mat, (indx), axis=0)

# we have some issues with overflow of 3rd col
    overflow_idx = np.where(thetas[:,2] < -300)
    thetas[overflow_idx, :] += np.array([0,0,2*179])

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
    #print('3D torques:\n', torques)
    #print('3D deflections: \n', thetas)
    #torquesX, torquesY, torquesZ = torques[:,0] , torques[:,1], torques[:,2]
    BigTheta = np.vstack((BigTheta, thetas))
    BigTorque = np.vstack((BigTorque, torques))

BigTheta = BigTheta[1:,:] #remove first row of zeros, from init
BigTorque = BigTorque[1:,:]
print(BigTheta.shape) #n.3
print(BigTorque.shape) #n.3

print('number of datapoints', BigTheta.shape)

#### CALCULATE K ####
matK = np.linalg.lstsq(BigTheta, BigTorque, rcond=None)[0]

print(matK.shape)
print(matK)
print('position number: ', i+1)

#===============================================
#### PLOT ####
#===============================================

# let us retrieve estimates of K via sklearn library to check numpy.lstsq, which gives us all zeros
# for the third row of K :(

names = ['x:roll', 'y:pitch', 'z:yaw']
dim = 1
torq_1D = BigTorque[:,dim] #z col
#theta_1D = BigTheta[:,dim]
print('torq1d shape', torq_1D.shape)

myX = BigTheta#theta_1Dreshape(-1,1)
myy = torq_1D 

#regr= Ridge(fit_intercept=True, alpha=1.0, random_state=0, normalize=True)
regr = linear_model.LinearRegression()
regr.fit(myX, myy)
coef_ridge = regr.coef_
#gridx = np.linspace(myX.min(), myX.max(), 20)
#coef_ = regr.coef_ * gridx + regr.intercept_
yPred= regr.predict(myX) 

print('K Coefficients: \n', regr.coef_)
print("Mean squared error: %.2f" % mean_squared_error(torq_1D, yPred)) 
#print('Variance score (ideal 1): %.2f' % r2_score(thetaY))

# http://stackabuse.com/linear-regression-in-python-with-scikit-learn/
'''
#===============================================
#### PLOT ####
#===============================================

# let us retrieve estimates of K via sklearn library to check numpy.lstsq, which gives us all zeros
# for the third row of K :(

## Note: For the IMU, orientation.Y is pitch; X is roll; Z is yaw

trace0 = go.Scatter( x = torq_1D, y = theta_1D, mode = 'markers',
    name = names[dim] + ' deflection in degrees (by IMU)' )

myX = theta_1D.reshape(-1,1)
myy = torq_1D 

#regr= Ridge(fit_intercept=True, alpha=1.0, random_state=0, normalize=True)
regr = linear_model.LinearRegression()
regr.fit(myX, myy)
coef_ridge = regr.coef_
gridx = np.linspace(myX.min(), myX.max(), 20)
coef_ = regr.coef_ * gridx + regr.intercept_
#print(regr.coef_)
#print(regr.intercept_)
yPred= regr.predict(myX) 
#plt.plot(gridx, coef_, 'g-', label="ridge regression")

trace2 = go.Scatter( x= gridx, y = coef_,
    name = 'linear fit (w/ridge penalty)' )

data = [trace0, trace2]
    
layout = go.Layout(
    title= names[dim] + ' degrees of deflection vs torque',
    yaxis=dict(title='torque (in grams)'),
    xaxis=dict(title='degrees'),
    legend=dict(x=.1, y=-.5) )

fig = go.Figure(data=data, layout=layout)

print('K Coefficients: \n', regr.coef_)
print("Mean squared error: %.2f" % mean_squared_error(theta_1D, yPred)) 
#print('Variance score (ideal 1): %.2f' % r2_score(thetaY))
po.plot(fig)

'''
