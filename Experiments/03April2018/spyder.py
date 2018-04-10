#>!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 00:08:30 2018

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
pos = range(1,16)
#IMUDats = [ '%02dIMU.txt'% x for x in pos ]
IMUCols = ['timeSysCal', 'XYZ','X', 'Y', 'Z']

listpos = range(15)
#listpos = [5]
BigTheta = []
BigTorque = []
for i in listpos:
    print(i)
    fname =  '%02dIMU.txt'%(i+1)
    imuDat = pd.read_csv(path+fname, header=None,sep=';', 
                        names = IMUCols, skip_blank_lines=True, usecols=[0,1,2,3,4]) #TODO: this is somehow skipping a col, so the last is NaN. usecols to get around this
    imuDat = imuDat.drop(['timeSysCal', 'XYZ'], 1)
    bkgd = imuDat.iloc[0::2]  
    signal = imuDat.iloc[1::2]
    zer = bkgd.as_matrix()
    sig = signal.as_matrix()
    print(zer)
    print(len(zer))
    print(len(sig))
    thetas = sig-zer
    thetaX, thetaY, thetaZ = thetas[:,0], thetas[:,1], thetas[:,2]
    xs = [4.6, 4.1, 3.5, 3.1, 2.6]#pos 1 x = 4.6 cm
    #ys = [0.4, 0.1, -0.2]
    ys = [0.4, 0, -0.2]
    n = thetas.shape[0]
    forces = range(1, n+1) 
    forcesZ = [20*f for f in forces]
    
    posX = np.array([(x,x,x) for x in xs]).flatten()
    posY = np.array(ys *5)
    posZ = np.array([0]*15)
    pos = np.array([posX[i], posY[i], posZ[i]])
    forcesXYZ = np.column_stack((np.zeros((n,2)),forcesZ))
    torques = np.cross(forcesXYZ, pos)
    torquesX, torquesY, torquesZ = torques[:,0] , torques[:,1], torques[:,2]
    #BigTorque.column_stack(np.array(torquesY))
    BigTorque.append(np.array(torquesY))
    BigTheta.append(np.array(thetaY))
    #BigTheta.column_stack(np.array(thetaY))
print('exit loop')
np.concatenate(BigTheta).ravel()
np.concatenate(BigTorque).ravel()
BigTheta = np.hstack(BigTheta)
BigTorque= np.hstack(BigTorque)

#%%
#po.init_noteb  le?ook_mode(connected=True)

torquesY = BigTorque
thetaY = BigTheta
#print(torquesY)
#print(thetaY)
trace0 = go.Scatter( x = torquesY, y = thetaY, mode = 'markers',
    name = 'degrees (by IMU)' )

#forcesXYZ = forcesXYZ.reshape(-1, 1)
myX = torquesY.reshape(-1,1)
myy = thetaY 
#print(len(thetas))
#regr= Ridge(fit_intercept=True, alpha=1.0, random_state=0, normalize=True)
regr = linear_model.LinearRegression()
regr.fit(myX, myy)
coef_ridge = regr.coef_
gridx = np.linspace(myX.min(), myX.max(), 20)
coef_ = regr.coef_ * gridx + regr.intercept_
print('test')
print(regr.coef_)
print(regr.intercept_)
yPred= regr.predict(myX) 
#plt.plot(gridx, coef_, 'g-', label="ridge regression")

trace2 = go.Scatter( x= gridx, y = coef_,
    name = 'linear fit (w/ridge penalty)' )

data = [trace0, trace2]
    
layout = go.Layout(
    title='torque vs Degrees of Deflection',
    yaxis=dict(title='degrees'),
    xaxis=dict(title='torque (in grams)'),
    legend=dict(x=.1, y=-.5) )

fig = go.Figure(data=data, layout=layout)
# Plot and embed in ipython notebook!

print('Coefficients: \n', regr.coef_)
print("Mean squared error: %.2f" % mean_squared_error(thetaY, yPred))
#print('Variance score (ideal 1): %.2f' % r2_score(thetaY))
po.plot(fig)

# =============================================================================
#     
# =============================================================================



