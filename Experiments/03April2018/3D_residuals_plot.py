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
import shelve

with shelve.open('calculated_data', 'r') as shelf:
    BigTheta = shelf['BigTheta']
    BigTorque = shelf['BigTorque']

#path = "~/Documents/projects_Spring2018/howe299r/Experiments/03April2018/WIP/"
IMUCols = ['timeSysCal', 'XYZ','X', 'Y', 'Z']

#===============================================
#### DECLARE CONSTANTS ####
#===============================================

print('number of datapoints', BigTheta.shape)

#### CALCULATE K ####
print('\n======================')
# matK = np.linalg.lstsq(BigTorque, BigTheta, rcond=None)[0]
# print(matK.shape)
# print('numpy lstsq K coefficients (numpy lstsq):\n', matK)

#===============================================
#### FIT TO ESTIMATE K ####
#===============================================

## Note: For the IMU, orientation.Y is pitch; X is roll; Z is yaw
torq_names = ['x', 'y', 'z']
dim = 1
torq_1d = BigTorque[:,dim] 
torq = BigTorque
theta = BigTheta 
print('torq shape', torq.shape)

myX = BigTheta#theta_1Dreshape(-1,1)
myy = torq 

#regr= Ridge(fit_intercept=False, alpha=1.0, random_state=0, normalize=True)
regr = linear_model.LinearRegression()
regr.fit(myX, myy)
K = regr.coef_
yPred= regr.predict(myX) 

print('\n======================')
print('K Coefficients: \n', K)
# #print('Variance score (ideal 1): %.2f' % r2_score(thetaY))
# print('\n======================')
# print('Mean Absolute Error:', metrics.mean_absolute_error(torq, yPred))  
# print('Mean Squared Error:', metrics.mean_squared_error(torq, yPred))  
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(torq, yPred)))
# print('\n======================')
# print('torques about y axis: Min', myy.min(), '; Max', myy.max(), 'grams * cm')
torq_est = np.dot(K, theta.T)
resid = torq - torq_est.T
mse = (resid ** 2).mean(axis=0)

print('\n======================')
print('resid shape', resid.shape)
print('mse shape', mse.shape)
print('rmse', np.sqrt(mse))
print('\n======================')

# torq_est[column 1] = torque y's



#===============================================
#### PLOT ####
#===============================================

print(resid)
print(resid.shape)
names = ['X', 'Y', 'Z']
param = 'Torque'
dim = 1

xplot = torq_est.T[:,dim]
xplot2 = torq[:,dim]
print(xplot.shape)
yplot = resid[:,dim] 
print(yplot.shape)

trace0 = go.Scatter( x = xplot, y = yplot, mode = 'markers',
    name = '%s-axis %s estimated'%(names[dim], param))

trace1 = go.Scatter( x = xplot2, y = yplot, mode = 'markers', 
name = '%s-axis %s calculated from data'%(names[dim], param))
# trace1 = go.Scatter( x = BigTheta[:,dim], y = torq, mode = 'markers',  name = torq_names[dim] + ' torque (from data), in g*cm (by IMU), using 3d K' ) 
# trace1 = go.Scatter( x = BigTheta[:,dim], y = torq, mode = 'markers',  name = torq_names[dim] + ' torque (from data), in g*cm (by IMU), using 3d K' ) 
data = [trace0, trace1]

layout = go.Layout(
    title='%s-axis %s: Resid vs Estimate (with 3x3 K, using SkLearn LinReg) (IMU data)' % (names[dim], param),
    yaxis=dict(title= 'resid (g cm)'),
    xaxis=dict(title='%s (g cm)' % param),
    legend=dict(x=.1, y=0.8) )

fig = go.Figure(data=data, layout=layout)

po.plot(fig)

