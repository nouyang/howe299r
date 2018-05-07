"""
Created on 07 May 2018
@author: nrw

Adds linear fit torque estimate XYZ, linear fit residuals XYZ to shelved data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn import metrics
import shelve
from sklearn.ensemble import RandomForestRegressor

with shelve.open('cleaned_data', 'r') as shelf:
    BigTheta = shelf['BigTheta']
    BigTorque = shelf['BigTorque']
    BigForce = shelf['BigForce'] 
    BigPosition = shelf['BigPosition'] 
    BigPosIdx = shelf['BigPosIdx']


#===============================================
#### DECLARE CONSTANTS ####
#===============================================

print('number of datapoints', BigTheta.shape)


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

### 
# --- Linear Fit
###

linReg = linear_model.LinearRegression(fit_intercept=False)
linReg.fit(myX, myy)
K_lin = linReg.coef_ 
dim = 1
#torq_1d = BigTorque[:,dim] 
linK = K_lin
yPredLin= linReg.predict(myX) 

### 
# --- Random Forest Fit
###
rfreg = RandomForestRegressor(max_depth=2, random_state=0)
rfreg.fit(myX, myy)
rfpred = rfreg.predict(myX)

print('\n======================')
print('Numpy linalg.lstsq() K coefficients:\n', matK)
print('LinReg K Coefficients: \n', K_lin)
print('\n======================')

#######################################
# --  torq_est Definition 
#######################################
torq_est = yPredLin

# --  quick sanity check 
resid = torq - torq_est 
print('Residual mean: ', resid.mean(axis=0), ', Std Dev: ', resid.std(axis=0))
print('Linear fit torq est mean: ', torq_est.mean(axis=0), ', Torq Est: ', torq_est.std(axis=0))

#print('Variance score (ideal 1): %.2f' % r2_score(thetaY))
#print('Mean Absolute Error: %0.02f' % metrics.mean_absolute_error(torq, yPred))  

print('\n=======  SkLearn Metrics====')
print('\n---- Using LinReg K dot theta:')
rmse = metrics.mean_squared_error(torq, torq_est, multioutput='raw_values')**0.5
print('Root Mean Squared Error: %s' % str(rmse))

print('\n---- Using  RF: ')
rmse = metrics.mean_squared_error(torq, rfpred, multioutput='raw_values')**0.5
print('Root Mean Squared Error: %s' % str(rmse))

print('\n---- Using sklearn LinearRegression.pred(theta).   ========')
rmse = metrics.mean_squared_error(torq, yPredLin, multioutput='raw_values')**0.5
stddev = resid.std(axis=0)
print('Root Mean Squared Error: %s' % str(rmse))
print('Population StdDev: %s' % str(stddev))


print('\nNote: torques about y axis: Min', myy.min(), '; Max', myy.max(), 'grams * cm')
print('\n======================')


full_data = np.hstack((BigPosition, BigForce, BigTheta, BigTorque))#, BigPosIdx))
full_data = np.hstack((full_data, torq_est, resid))
full_data[:,-1] = BigPosIdx
print(torq_est.shape)
print(resid.shape)
print(full_data.shape)

f=open('full_calculated_data.csv', 'w')
f.write('Big Pos, , , Big Force, , , BigTheta, , , BigTorque, , ,  \
    Torq Est, , , Resid, , BigPosIdx\n')
f.write('X, Y, Z,'*6 + ' \n')
f.close()
f=open('full_calculated_data.csv', 'a')
np.savetxt(f, full_data, delimiter=",", fmt='%0.02f')

with shelve.open('resid_data', 'c') as shelf:
    shelf['torq_est'] = torq_est
    shelf['resid'] = resid
    shelf['K'] = K_lin #linear fit
