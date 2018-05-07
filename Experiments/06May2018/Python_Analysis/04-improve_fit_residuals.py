"""
Calculate residuals and fit with linear & RF models
Display fit to residuals
Collect statistics about new compensated model
Do cross validation
Created May 2018
@author: nrw
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shelve
from datetime import datetime

from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn import metrics

from sklearn.ensemble import RandomForestRegressor

import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

sns.set()


#===============================================
#### DECLARE CONSTANTS ####
#===============================================

with shelve.open('calculated_data', 'r') as shelf:
    BigTheta = shelf['BigTheta']
    BigTorque = shelf['BigTorque']
    BigForce = shelf['BigForce'] 
    BigPosition = shelf['BigPosition'] 
    BigPosIdx = shelf['BigPosIdx'] 

with shelve.open('calculated_data2', 'r') as shelf:
    torq_est = shelf['torq_est'] 
    resid = shelf['resid'] 
    K = shelf['K'] 


#===============================================
#### DECLARE CONSTANTS ####
#===============================================

# using K as calculated from linear regression
torq_est = np.dot(K, BigTheta.T).T #n.3
# print(torq_est)
# print(BigTorque)
resids = torq_est - BigTorque
# print(resids)
print(np.average(resids[:,1]))
print(np.average(resids[:,1]**2)**0.5)
# add residuals
myX = BigTheta
myy = resids
linreg = linear_model.LinearRegression() #allow y intercpt
linreg.fit(myX, myy)
# rfreg = RandomForestRegressor(max_depth=2, random_state=0)
# rfreg.fit(myX, myy.ravel())
# K = linregr.coef_
linresidpredict= linreg.predict(myX) 
# rfpredict= rfreg.predict(myX) 

rfreg = RandomForestRegressor(max_depth=2, random_state=0)
rfreg.fit(myX, myy)
rfresidpredict= rfreg.predict(myX) 


torq_corrected = torq_est - linresidpredict
resids_corrected = torq_corrected - BigTorque
print('\n')
print(np.average(resids_corrected[:,1]))
print(np.average(resids_corrected[:,1]**2)**0.5)

torq_corrected = torq_est - rfresidpredict
resids_corrected = torq_corrected - BigTorque
print('\n')
print(np.average(resids_corrected[:,1]))
print(np.average(resids_corrected[:,1]**2)**0.5)
#===============================================
#### DECLARE CONSTANTS ####
#===============================================

print('\n---- Using  RF: ')
rfrmse = metrics.mean_squared_error(torq, rfpred, multioutput='raw_values')**0.5
print('Root Mean Squared Error: %s' % str(rfrmse))
