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
from sklearn import metrics

path = "~/Documents/projects_Spring2018/howe299r/Experiments/03April2018/WIP/"
#IMUDats = [ '%02dIMU.txt'% x for x in pos ]
IMUCols = ['timeSysCal', 'XYZ','X', 'Y', 'Z']

BigTheta = np.zeros((1,3))
BigTorque = np.zeros((1,3))


#===============================================
#### DECLARE CONSTANTS ####
#===============================================
listpos = range(15)
listpos = [2]
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
    #print('dataframe stats')
    #print(imuDat.describe())
    #print(imuDat.head())
    bkgd = imuDat.iloc[0::2]  
    signal = imuDat.iloc[1::2]
    zer = bkgd.as_matrix()
    sig = signal.as_matrix()

    #### DECLARE CONSTANTS ####
    pos = np.array([posX[i], posY[i], posZ[i]])
    thetas = sig-zer
    print('thetas\n', thetas)
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
    print('torques\n', torques)
    #torques = np.cross(forces, pos.reshape(-1,1).T) #n.3
    #print('3D torques:\n', torques)
    #print('3D deflections: \n', thetas)
    #torquesX, torquesY, torquesZ = torques[:,0] , torques[:,1], torques[:,2]
    BigTheta = np.vstack((BigTheta, thetas))
    BigTorque = np.vstack((BigTorque, torques))

BigTheta = BigTheta[1:,:] #remove first row of zeros, from init
BigTorque = BigTorque[1:,:]
print(BigTheta.shape) #n.3
print(BigTorque) #n.3

print('number of datapoints', BigTheta.shape)

#### CALCULATE K ####
matK = np.linalg.lstsq(BigTorque, BigTheta, rcond=None)[0]
# a = BigTheta.T
# b = BigTorque
# SOL = np.dot(np.dot(np.linalg.inv(np.dot(a.T,a)),a.T),b)
# print(SOL.shape)
print(matK.shape)
print('K coefficients (numply lstsq):\n', matK)
print('position number: ', i+1)

#===============================================
#### FIT TO ESTIMATE K ####
#===============================================

# let us retrieve estimates of K via sklearn library to check numpy.lstsq, which gives us all zeros
# for the third row of K :(

## Note: For the IMU, orientation.Y is pitch; X is roll; Z is yaw
torq_names = ['x', 'y', 'z']
dim = 1
torq_1D = BigTorque[:,dim] 
torq_1D = BigTorque
#theta_1D = BigTheta[:,dim]
print('torq1d shape', torq_1D.shape)

myX = BigTheta#theta_1Dreshape(-1,1)
myy = torq_1D 

#regr= Ridge(fit_intercept=False, alpha=1.0, random_state=0, normalize=True)
regr = linear_model.LinearRegression()
regr.fit(myX, myy)
coef_ridge = regr.coef_
K = regr.coef_
#gridx = np.linspace(myX.min(), myX.max(), 20)
#coef_ = regr.coef_ * gridx + regr.intercept_
yPred= regr.predict(myX) 

print('\n======================')
print('K Coefficients: \n', K)
#print("Mean squared error: %.2f" % mean_squared_error(torq_1D, yPred)) 
#print('Variance score (ideal 1): %.2f' % r2_score(thetaY))
print('\n======================')
print('Mean Absolute Error:', metrics.mean_absolute_error(torq_1D, yPred))  
print('Mean Squared Error:', metrics.mean_squared_error(torq_1D, yPred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(torq_1D, yPred)))
print('\n======================')
print('torques about y axis: Min', myy.min(), '; Max', myy.max(), 'grams * cm')

#gridx = np.linspace(myX.min(), myX.max(), 20)

# http://stackabuse.com/linear-regression-in-python-with-scikit-learn/

#===============================================
#### PLOT ####
#===============================================


trace0 = go.Scatter( x = BigTheta[:,dim], y = yPred, mode = 'markers',
    name = torq_names[dim] + ' torque (predicted), in g*cm (by IMU), using 3d K' )
trace1 = go.Scatter( x = BigTheta[:,dim], y = torq_1D, mode = 'markers',
    name = torq_names[dim] + ' torque (from data), in g*cm (by IMU), using 3d K' )

# coef_ridge = regr.coef_
# gridx = np.linspace(myX.min(), myX.max(), 20)
#coef_ = regr.coef_ * gridx + regr.intercept_

#plt.plot(gridx, coef_, 'g-', label="ridge regression")

# trace2 = go.Scatter( x= gridx, y = coef_,
    # name = 'linear fit (w/ridge penalty)' )

data = [trace0, trace1]
K_str = ['%.02f' % x for x in K]
K_str = ', '.join(K_str)
    
layout = go.Layout(
    title= 'Ridge Regr, pitch (up/down) degrees of deflection, vs %s torque <br>K: %s' % (torq_names[dim], K_str),
    yaxis=dict(title= torq_names[dim] + 'axis torque (in grams cm)'),
    xaxis=dict(title='pitch degrees'),
    legend=dict(x=.1, y=0.8) )

fig = go.Figure(data=data, layout=layout)

#print("Mean squared error: %.2f" % metrics.mean_squared_error(torq_1D, yPred)) 
#print('Variance score (ideal 1): %.2f' % r2_score(thetaY))
po.plot(fig)

