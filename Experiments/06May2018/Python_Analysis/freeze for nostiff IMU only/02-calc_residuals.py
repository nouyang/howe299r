"""
Created on 07 May 2018
@author: nrw

Calculates linear fit and plots torque estimate
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

import seaborn as sns
sns.set()

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
print('LinReg K Coefficients: \n', K_lin)
print('\n======================')

#######################################
# --  torq_est Definition 
#######################################
torq_est = yPredLin

# --  quick sanity check 
resid = torq - torq_est 
print('Residual mean: ', resid.mean(axis=0))
print('Residual Std Dev: ', resid.std(axis=0))
print('Linear fit torq est mean: ', torq_est.mean(axis=0))
print('Linear fit torq Est Std Dev: ', torq_est.std(axis=0))

#print('Variance score (ideal 1): %.2f' % r2_score(thetaY))
#print('Mean Absolute Error: %0.02f' % metrics.mean_absolute_error(torq, yPred))  

#######################################
# -- Fit metrics 
#######################################
print('\n=======  SkLearn Metrics====')
print('\n---- Using  RF: ')
rfrmse = metrics.mean_squared_error(torq, rfpred, multioutput='raw_values')**0.5
print('Root Mean Squared Error: %s' % str(rfrmse))

print('\n---- Using sklearn LinearRegression.pred(theta).   ========')
linrmse = metrics.mean_squared_error(torq, yPredLin, multioutput='raw_values')**0.5
stddev = resid.std(axis=0)
print('Root Mean Squared Error: %s' % str(linrmse))
print('Population StdDev: %s' % str(stddev))


print('\nNote: torques about y axis: Min', myy.min(), '; Max', myy.max(), 'grams * cm')
print('\n======================')


#######################################
# -- Store data 
#######################################
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

#######################################
# -- Graph linear fit 
#######################################
df=pd.DataFrame({'Resid of Lin TorqX fit':resid[:,0],
                 'Resid of Lin TorqY fit':resid[:,1],
                 'TorqX measured (g*cm)':BigTorque[:,0],
                 'TorqY measured (g*cm)':BigTorque[:,1],
                 'TorqX linear estimated (g*cm)': torq_est[:,0],
                 'TorqY linear estimated (g*cm)': torq_est[:,1],
                 'TorqX rf estimated (g*cm)': rfpred[:,0],
                 'TorqY rf estimated (g*cm)': rfpred[:,1],
                 'Theta Y (deg)': BigTheta[:,1]
                 })



# ----------------------- Plot pair (or more) of plots
# these two graphs don't make sense, because RF is not a kind='ref' linear fit graph
# sns.pairplot(data=df, kind='scatter', y_vars=['TorqY linear estimated (g*cm)', 'TorqY measured (g*cm)'],
# sns.pairplot(data=df, kind='reg', y_vars=['TorqY linear estimated (g*cm)', 'TorqY rf estimated (g*cm)'],
             # x_vars=['Theta Y (deg)'])

# plt.scatter(df['Theta Y (deg)'], df['TorqY linear estimated (g*cm)'], label='Lin fit, RMSE: %0.2f'% linrmse[1])
# plt.scatter(df['Theta Y (deg)'], df['TorqY rf estimated (g*cm)'], label='RF fit, RMSE: %0.2f'% rfrmse[1])
# plt.scatter(df['Theta Y (deg)'], df['TorqY measured (g*cm)'], label='Measured data') 
# plt.legend()
# plt.title('Fitting models to torque Y data')
# plt.xlabel('Theta Y (deg)')
# plt.ylabel('Torque Y (g*cm)')

sns.lmplot(data=df, y='TorqY measured (g*cm)', x='Theta Y (deg)')
plt.show()



# Matplotlib color cycle example
# from itertools import cycle
# import numpy as np

# import matplotlib.pyplot as plt

# color_gen = cycle(('blue', 'lightgreen', 'red', 'purple', 'gray', 'cyan'))

# for lab in np.unique(df['dataset']):
    # plt.scatter(df.loc[df['dataset'] == lab, 'x'], 
                # df.loc[df['dataset'] == lab, 'y'], 
                # c=next(color_gen),
                # label=lab)

# plt.legend(loc='best')
