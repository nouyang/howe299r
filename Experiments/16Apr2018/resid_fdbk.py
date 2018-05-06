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

axes = ['x', 'y', 'z']
dim = 1

myX = BigTheta[:,1].reshape(-1,1)
#myX = BigTheta
myy = resid[:,1].reshape(-1,1)
print(BigTheta.shape)


linreg = linear_model.LinearRegression() #allow y intercpt
linreg.fit(myX, myy)
rfreg = RandomForestRegressor(max_depth=2, random_state=0)
rfreg.fit(myX, myy.ravel())
#K = linregr.coef_
linpredict= linreg.predict(myX) 
rfpredict= rfreg.predict(myX) 

# X_train, X_test, y_train, y_test = train_test_split(
    # ...     iris.data, iris.target, test_size=0.4, random_state=0)


linrmse = metrics.mean_squared_error(myy, linpredict, multioutput='raw_values')**0.5
print('lin rmse: %0.3f' % linrmse)
print(linreg.score(myX, myy))
rfrmse = metrics.mean_squared_error(myy, rfpredict, multioutput='raw_values')**0.5
print('rf rmse: %0.3f' % rfrmse)

scores = cross_val_score(linreg, myX, myy, cv=5)
# print('lin scores (5 fold cross validation): %s' % str(scores)) #TODO: why are the cross validation scores so low??
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# print('\n')

scores = cross_val_score(rfreg, myX, myy.ravel(), cv=5)
# print('rf scores (5 fold cross validation): %s' % str(scores))
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


### do some quick and dirty train test check
# X_train, X_test, y_train, y_test = train_test_split(
    # myX, myy, test_size=0.33, random_state=42)

xsplit = np.array_split(myX, [63], axis=0)
ysplit = np.array_split(myy, [63], axis=0)
X_train, X_test = xsplit[1], xsplit[0] #test aaginst 1st x position
y_train, y_test = ysplit[1], ysplit[0]

linreg = linear_model.LinearRegression() #allow y intercpt
linreg.fit(X_train, y_train)
rfreg = RandomForestRegressor(max_depth=2, random_state=0)
rfreg.fit(X_train, y_train.ravel())

linSPLITpredict= linreg.predict(X_test)
rfSPLITpredict= rfreg.predict(X_test) 

linrmse = metrics.mean_squared_error(y_test, linSPLITpredict, multioutput='raw_values')**0.5
print('lin test rmse: %0.3f' % linrmse)
print('lin train rmse: %0.3f' % linrmse)
rfrmse = metrics.mean_squared_error(y_test, rfSPLITpredict, multioutput='raw_values')**0.5

print('rf test rmse: %0.3f' % rfrmse)
print('rf train rmse: %0.3f' % rfrmse)

# linrmse = metrics.mean_squared_error(y_test, linSPLITpredict, multioutput='raw_values')**0.5
# print('lin rmse: %0.3f' % linrmse)
# rfrmse = metrics.mean_squared_error(y_test, rfSPLITpredict, multioutput='raw_values')**0.5
# print('rf rmse: %0.3f' % rfrmse)

#===============================================
#### plot fits ####
#===============================================

# print('~~~~~~~~~~Plotting!' + '~~~~~~~~')

df=pd.DataFrame({'Measured Theta Y (deg)' :BigTheta[:,1],
                 'True Torque Y Resid (g*cm) (linear model)': resid[:,1], 
                 'Lin Reg Resid Fit': linpredict.ravel(),
                 'Random Forests Resid Fit': rfpredict.ravel()
                 })
                 # 'Colors':colorsIdx})

# sns.pairplot(data=df, y_vars=['Measured Theta Y (deg)'], x_vars=[
                 # 'True Torque Y Resid (g*cm) (linear model)',
                 # 'Lin Reg Resid Fit',
                 # 'Random Forests Resid Fit'])
plt.scatter(X_train, y_train, label='training data', alpha=0.5) 
plt.scatter(X_test, y_test, label='test using 1st x pos') 
plt.scatter(X_test, linSPLITpredict.ravel(), label='lin') 
plt.scatter(X_test, rfSPLITpredict.ravel(), label='rf')
plt.legend()
plt.ylabel('Resid (g cm)')
plt.ylabel('Theta')
plt.suptitle('Fitted residuals')
strtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
ax = plt.gca()
plt.text(1.1, 0, 'Time: '+strtime, horizontalalignment='left', verticalalignment='bottom',
        transform = ax.transAxes, fontsize=6)
plt.show()
