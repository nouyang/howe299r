"""
Created on Thu Apr 16
@author: nrw
This plots residuals, 
And also takes shelved torque data, adds in torque estimate and residual data
And writes it all to a CSV (and also to a second shelf file)
16 Apr: Fixing resid and torque estimate calculations (y intercept issue)
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
    BigPosIdx = shelf['BigPosIdx']

#path = "~/Documents/projects_Spring2018/howe299r/Experiments/03April2018/WIP/"
IMUCols = ['timeSysCal', 'XYZ','X', 'Y', 'Z']

#===============================================
#### DECLARE CONSTANTS ####
#===============================================

print('number of datapoints', BigTheta.shape)

#### CALCULATE K ####

#===============================================
#### FIT TO ESTIMATE K ####
#===============================================

## Note: For the IMU, orientation.Y is pitch; X is roll; Z is yaw
torq_names = ['x', 'y', 'z']
dim = 1
#torq_1d = BigTorque[:,dim] 
torq = BigTorque
theta = BigTheta 
print('torq shape', torq.shape)

myX = BigTheta#theta_1Dreshape(-1,1)
myy = torq 

regr= Ridge(fit_intercept=False, alpha=1.0, random_state=0, normalize=True) #TODO how does fitting yintercept make rmse worse?
regr2 = linear_model.LinearRegression(fit_intercept=False)
regr.fit(myX, myy)
regr2.fit(myX, myy)
K = regr.coef_
K2 = regr2.coef_ 
yPred= regr.predict(myX) 
yPred2= regr2.predict(myX) 

print('\n======================')
matK = np.linalg.lstsq(BigTorque, BigTheta, rcond=None)[0]
print(matK.shape)
print('Numpy linalg.lstsq() K coefficients:\n', matK)
print('LinReg K Coefficients: \n', K2)
print('Ridge K Coefficients: \n', K)
print('\n======================')

#######################################
# --  torq_est Definition 
#######################################
torq_est = yPred2
torq_est2 = np.dot(matK, theta.T).T #n.3

# --  quick sanity check 
# resid = torq - torq_est 
# # resid2 = torq - torq_est2
# mse = (resid ** 2).mean(axis=0)
# print('resid shape', resid.shape)
# print('RMSE Per Torque Dim', np.sqrt(mse))

#print('Variance score (ideal 1): %.2f' % r2_score(thetaY))
#print('Mean Absolute Error: %0.02f' % metrics.mean_absolute_error(torq, yPred))  

print('\n=======  SkLearn Metrics====')
print('\n---- Using LinReg K dot theta. This has worse error as we have no intercept term. ===')
rmse = metrics.mean_squared_error(torq, torq_est2, multioutput='raw_values')**0.5
print('Root Mean Squared Error: %s' % str(rmse))

print('\n---- Using sklearn LinearRegression.pred(theta).   ========')
rmse = metrics.mean_squared_error(torq, yPred2, multioutput='raw_values')**0.5
print('Root Mean Squared Error: %s' % str(rmse))

print('\n---- Using sklearn Ridge.pred(theta).   ========')
rmse = metrics.mean_squared_error(torq, yPred, multioutput='raw_values')**0.5
print('Root Mean Squared Error: %s' % str(rmse))
print('\n --- LinRegr and Ridge have same the best fit after removing intercept----')

print('\nNote: torques about y axis: Min', myy.min(), '; Max', myy.max(), 'grams * cm')
print('\n======================')


full_data = np.hstack((BigPosition, BigForce, BigTheta, BigTorque, BigPosIdx))


full_data = np.hstack((full_data, torq_est, resid))
print(torq_est.shape)
print(resid.shape)
np.savetxt("full_calculated_data.csv", full_data, delimiter=",", fmt='%0.02f')

with shelve.open('calculated_data2', 'c') as shelf:
    shelf['torq_est'] = torq_est
    shelf['resid'] = resid
    shelf['K'] = K
# #===============================================
# #### PLOT: Residuals (of Y torque_est - torque) vs Force (Z only)
# #===============================================

# print(resid.shape)
# names = ['X', 'Y', 'Z']
# param = 'Torque'
# x2param = 'Force'
# dim = 0

# xplot = torq_est[:,dim]
# xplot2 = BigForce[:,2]
# yplot = resid[:,dim] 

# trace0 = go.Scatter( x = xplot, y = yplot, mode = 'markers',
    # name = 'resid_torqY vs %s-axis %s estimated'%(names[dim], param))

# trace1 = go.Scatter( x = xplot2, y = yplot, mode = 'markers', 
# name = 'resid_torqY vs Resid vs Z-axis Force, as applied')

# #data = [trace0]

# overall_title='%s-axis %s: Resid vs Force applied (with 3x3 K, using SkLearn LinReg) (IMU data)' % \
    # (names[dim], param) + '<br>K: ' + np.array_str(K, precision=2) + '<br>'

# yaxistitle= 'resid (g cm)'
# xaxistitle= 'force (g)'

# layout = go.Layout(
    # title = overall_title,
    # legend=dict(x=.5, y=0.1) )

# fig = tools.make_subplots(rows=2, cols=1, subplot_titles=(trace0.name, trace1.name))

# fig.append_trace(trace0, 1,1)
# fig.append_trace(trace1, 2,1)

# fig['layout'].update(title=overall_title, showlegend=False)
# fig['layout']['xaxis1'].update(title='%s torque est (g cm)' % (names[dim]))
# fig['layout']['xaxis2'].update(title=xaxistitle)
# fig['layout']['yaxis1'].update(title=yaxistitle)
# fig['layout']['yaxis2'].update(title=yaxistitle)

# #fig = go.Figure(data=data, layout=layout)
# #po.plot(fig)

