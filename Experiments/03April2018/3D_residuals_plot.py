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
from plotly import tools

from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn import metrics
import shelve

with shelve.open('calculated_data', 'r') as shelf:
    BigTheta = shelf['BigTheta']
    BigTorque = shelf['BigTorque']
    BigForce = shelf['BigForce'] 
    BigPosition = shelf['BigPosition'] 

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


'''
#===============================================
#### PLOT: Residuals (of Y torque_est - torque) vs Torque_est (either Y or X axis)
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
data = [trace0]

layout = go.Layout(
    title='%s-axis %s: Resid vs Estimate (with 3x3 K, using SkLearn LinReg) (IMU data)' % (names[dim], param),
    yaxis=dict(title= 'resid (g cm)'),
    xaxis=dict(title='%s (g cm)' % param),
    legend=dict(x=.5, y=0.1) )

fig = tools.make_subplots(rows=2, cols=1, subplot_titles=(trace1.name, trace0.name))

fig.append_trace(trace0, 1,1)
fig.append_trace(trace1, 2,1)
fig['layout'].update(title = layout.title)
fig['layout']['xaxis2'].update(title=layout.xaxis['title'])
fig['layout']['yaxis1'].update(title = layout.yaxis['title'])
fig['layout']['yaxis2'].update(title = layout.yaxis['title'])

#fig = go.Figure(data=data, layout=layout)

po.plot(fig)
'''

#===============================================
#### PLOT: Residuals (of Y torque_est - torque) vs Force (Z only)
#===============================================

print(resid.shape)
names = ['X', 'Y', 'Z']
param = 'Torque'
x2param = 'Force'
dim = 0

xplot = torq_est.T[:,dim]
xplot2 = BigForce[:,2]
yplot = resid[:,dim] 

trace0 = go.Scatter( x = xplot, y = yplot, mode = 'markers',
    name = 'resid_torqY vs %s-axis %s estimated'%(names[dim], param))

trace1 = go.Scatter( x = xplot2, y = yplot, mode = 'markers', 
name = 'resid_torqY vs Resid vs Z-axis Force, as applied')

#data = [trace0]

overall_title='%s-axis %s: Resid vs Force applied (with 3x3 K, using SkLearn LinReg) (IMU data)' % \
    (names[dim], param) + '<br>K: ' + np.array_str(K, precision=2) + '<br>'

yaxistitle= 'resid (g cm)'
xaxistitle= 'force (g)'

layout = go.Layout(
    title = overall_title,
    legend=dict(x=.5, y=0.1) )

fig = tools.make_subplots(rows=2, cols=1, subplot_titles=(trace0.name, trace1.name))

fig.append_trace(trace0, 1,1)
fig.append_trace(trace1, 2,1)

fig['layout'].update(title=overall_title, showlegend=False)
fig['layout']['xaxis1'].update(title='%s torque est (g cm)' % (names[dim]))
fig['layout']['xaxis2'].update(title=xaxistitle)
fig['layout']['yaxis1'].update(title=yaxistitle)
fig['layout']['yaxis2'].update(title=yaxistitle)

#fig = go.Figure(data=data, layout=layout)

po.plot(fig)
