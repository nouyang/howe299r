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
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

sns.set()

rng = np.random.RandomState(1)

sns.set_context("talk")
#===============================================
#### DECLARE CONSTANTS ####
#===============================================

posX1 = 'X pos1 = 4.6 cm'     
posX2 = 'X pos2 = 4.1 cm'     
posX3 = 'X pos3 = 3.5 cm'     
posX4 = 'X pos4 = 3.1 cm'     
posX5 = 'X pos5 = 2.6 cm'     

posY1 = 'Y pos1 = 0.2 cm'
posY2 = 'Y pos2 = -0.1 cm'
posY3 = 'Y pos3 = -0.4 cm' 

# ----------------- Comment / uncomment colorsList to get x pos and y pos hues  --
colorsList = [posX1, posX1, posX1, 
                 posX2, posX2, posX2, 
                 posX3, posX3, posX3, 
                 posX4, posX4, posX4, 
                 posX5, posX5, posX5]

colorsList = [posY1, posY1, posY1, posY1, posY1,
                posY2, posY2, posY2, posY2, posY2,
                posY3, posY3, posY3, posY3, posY3]


#===============================================
#### Import data ####
#===============================================

    # ---- not loaded 

infile = 'cleaned_data_no_tendon'
infile2 = 'resid_no_tendon'
# strdatatxt = 'No tendon load'

with shelve.open(infile, 'r') as shelf:
    BigTheta = shelf['BigTheta']
    BigTorque = shelf['BigTorque']
    BigForce = shelf['BigForce'] 
    BigPosition = shelf['BigPosition'] 
    BigPosIdx = shelf['BigPosIdx'] 

with shelve.open(infile2, 'r') as shelf:
    torq_est = shelf['torq_est'] 
    resid = shelf['resid'] 
    K = shelf['K'] 

colorsIdx =  [colorsList[i] for i in BigPosIdx]


df=pd.DataFrame({ 'TorqX measured (g*cm)':BigTorque[:,0],
                 'TorqY measured (g*cm)':BigTorque[:,1],
                 'ThetaX (deg)': BigTheta[:,0],
                 'ThetaY (deg)': BigTheta[:,1],
                 'ThetaZ (deg)': BigTheta[:,2],
                 'TorqX estimated (g*cm)': torq_est[:,0],
                 'TorqY estimated (g*cm)': torq_est[:,1],
                 'Colors':colorsIdx})


    # ---- loaded 
infile = 'cleaned_data_loaded_tendon'
infile2 = 'resid_loaded_tendon'
# strdatatxt = 'Loaded tendon'

with shelve.open(infile, 'r') as shelf:
    BigTheta = shelf['BigTheta']
    BigTorque = shelf['BigTorque']
    BigForce = shelf['BigForce'] 
    BigPosition = shelf['BigPosition'] 
    BigPosIdx = shelf['BigPosIdx'] 

with shelve.open(infile2, 'r') as shelf:
    torq_est = shelf['torq_est'] 
    resid = shelf['resid'] 
    K = shelf['K'] 


colorsIdx =  [colorsList[i] for i in BigPosIdx]


df2=pd.DataFrame({
                 'TorqX measured (g*cm)':BigTorque[:,0],
                 'TorqY measured (g*cm)':BigTorque[:,1],
                 'ThetaX (deg)': BigTheta[:,0],
                 'ThetaY (deg)': BigTheta[:,1],
                 'ThetaZ (deg)': BigTheta[:,2],
                 'TorqX estimated (g*cm)': torq_est[:,0],
                 'TorqY estimated (g*cm)': torq_est[:,1],
                 'Colors':colorsIdx})
#===============================================
#### Evaluate direct (non linear) etc. regression ####
# Adaboost on decision tree
# Randon forest 
# Gradiant Boostin Regressor 
#===============================================
orange='#be5e00'

myX = torq_est 
myy = BigTheta[:,1]#BigTheta
#myY = Residuals 

#======= Adaboost on decision tree

###############################################################
#### Ada Boost Fit of Resid 
###############################################################
ada = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                                                     n_estimators=300, random_state=rng)
ada.fit(myX, myy)
y_est= ada.predict(myX)

rmse = metrics.mean_squared_error(myy, y_est, multioutput='raw_values')**0.5
print(rmse)

plt.scatter(myX[:,1], myy, label='measured torque y', color='orange', s=20)
plt.scatter(myX[:,1], y_est, label='ADA estimated torque y', linewidth=1, alpha=0.6, color='k')
plt.xlabel("theta y (deg)")
plt.ylabel("torque y (g*cm)")
plt.title("Boosted Decision Tree Regression \ndirectly on data (no linear model)\n" 
          + 'RMSE: %.04f' % float(rmse))
plt.legend()
plt.show()

#======= Gradient Boosting Regressor 
###############################################################
#### Gradeint Boost Fit of Resid 
###############################################################
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                    max_depth=1, random_state=0, loss='ls')
gb.fit(myX, myy)
y_est= gb.predict(myX)

rmse = metrics.mean_squared_error(myy, y_est, multioutput='raw_values')**0.5
print(rmse)

plt.plot()
plt.scatter(myX[:,1], myy, label='measured torque y', color='orange', s=15)
plt.scatter(myX[:,1], y_est, label='GB estimated torque y', linewidth=1, alpha=0.6, color='b')
plt.xlabel("theta y (deg)")
plt.ylabel("torque y (g*cm)")
plt.title("Gradient Boost Regression \ndirectly on data (no linear model)\n" 
          + 'RMSE: %.04f' % float(rmse))
plt.legend()
plt.show()


###############################################################
#### RF Fit of Resid 
###############################################################
rf= RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
rf.fit(myX, myy)
y_est= rf.predict(myX) 

rmse = metrics.mean_squared_error(myy, y_est, multioutput='raw_values')**0.5
print(rmse)

plt.scatter(myX[:,1], myy, label='measured torque y', color='orange', s=15)
plt.scatter(myX[:,1], y_est, label='estimated torque y', linewidth=1, alpha=0.6, color='g')
plt.xlabel("theta y (deg)")
plt.ylabel("torque y (g*cm)")
plt.title("Random Forest Regression \ndirectly on data (no linear model)\n" 
          + 'RMSE: %.04f' % float(rmse))
plt.legend()
plt.show()

###############################################################
#### Linear Fit of Resid 
###############################################################
lin= LinearRegression(fit_intercept=False)
lin.fit(myX, myy)
y_est= lin.predict(myX) 

rmse = metrics.mean_squared_error(myy, y_est, multioutput='raw_values')**0.5
print(rmse)

plt.scatter(myX[:,1], myy, label='measured torque y', color='orange', s=15)
plt.scatter(myX[:,1], y_est, label='estimated torque y', linewidth=1, alpha=0.6, color='r')
plt.xlabel("theta y (deg)")
plt.ylabel("torque y (g*cm)")
plt.title("Linear Regression directly on data\n" 
          + 'RMSE: %.04f' % float(rmse))
plt.legend()
plt.show()

###############################################################
#### Evaluate linear K + linear fit on residuals 
###############################################################

# BigTheta = BigTheta[:,1].reshape(-1,1)
# BigTorque = BigTorque[:,1].reshape(-1,1)
# torq_est = torq_est[:,1].reshape(-1,1)

myX = BigTheta
myy = BigTorque
lin = linear_model.LinearRegression() #allow y intercpt
lin.fit(myX, myy)
y_est = lin.predict(myX)
rmse = metrics.mean_squared_error(BigTorque, y_est, multioutput='raw_values')**0.5
print('Pure linear fit, rmse: ', np.array_str(rmse, precision=2))
# torq_est = np.dot(K, BigTheta.T).T #n.3

#===--------------------------
# resids =  resid #shelved data
resids = resid.reshape(-1,1)
#===-------------------------
# print(resids)
#===
#### 2nd linear term , with intercept

#####################################################################
#### Attempting Linear Correction
#####################################################################
plt.plot()

# add residuals
myX = BigTheta
print(resids.shape)
myy = resids[]
# print(myy)
lin2 = linear_model.LinearRegression() #allow y intercpt
lin2.fit(myX, resids)
lin2_y_est = lin2.predict(myX) 
# print(lin_y_est)

rmse2 = metrics.mean_squared_error(resids, lin2_y_est, multioutput='raw_values')**0.5
print('sanity rmse on resids', np.array_str(rmse2, precision=2))

torq_corrected = np.array(y_est) + np.array(lin2_y_est) #resids
print('shape!!',torq_corrected.shape)
# print('shape!!',torq_est.shape)
print('lin2 est reid  shape!!',lin2_y_est.shape)
#lin2_y_est
# resids_corrected = torq_corrected - BigTorque
print('bigtorq shape', BigTorque.shape)
# print(torq_est[0:5,:])
# print(resids[0:5,:])
# print(lin2_y_est[0:5,:])
print(torq_corrected[0:5,:])
# print(resids_corrected[0:5,:])
rmse = metrics.mean_squared_error(BigTorque, torq_corrected, multioutput='raw_values')**0.5
print('Resid correct Fit with 2nd term (w/ intercept)\n -- rmse: ', np.array_str(rmse, precision=2))


# plt.scatter(myX, myy, label='resids y', color='orange', s=20)
# plt.plot(myX, lin2_y_est, label='est resids y', linewidth=1, alpha=0.6, color='c')
# plt.title("lin resids fit\n" + 'RMSE:' +str(rmse))
# plt.legend()
# plt.show()

# plt.scatter(myX, BigTorque, label='original torq', color='orange', s=20)
# plt.plot(myX, torq_corrected, label='torq estimate (lin corrected)', linewidth=1, alpha=0.6, color='r')
# plt.title("measured Torq vs new torq est (lin adjusted), " + 'RMSE: ' + str(rmse))
# plt.legend()
# plt.show()


#####################################################################
#### Attempting Random Forest correction
##################################################################### 
myX = BigTheta
myy = resids
rf = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
rf.fit(myX, myy)
rf_y_est= rf.predict(myX).reshape(-1,1)

rmse3 = metrics.mean_squared_error(resids, rf_y_est, multioutput='raw_values')**0.5
print('sanity rmse on randForest resids', np.array_str(rmse3, precision=2))

# plt.scatter(myX, myy, label='resids y', color='orange', s=20)
# plt.plot(myX, rf_y_est, label='est resids y', linewidth=1, alpha=0.6, color='c')
# plt.title("rf resids fit\n" + 'RMSE: ' + str(rmse3))
# plt.legend()
# plt.show()


torq_corrected = np.array(torq_est) + np.array(rf_y_est)
print('shape11!!',torq_corrected.shape)
# print('shape11!!',torq_est.shape)
# print('shape11!!',rf_y_est.shape)
# resids_corrected = np.array(torq_corrected)-np.array(BigTorque)

print('shape11!!',BigTorque.shape)
# print('shape11!!',resids_corrected.shape)
rmse = metrics.mean_squared_error(BigTorque[:,1], torq_corrected, multioutput='raw_values')**0.5
print('Resid corrected with random forest\n - rmse: ', np.array_str(rmse, precision=2))

plt.scatter(myX, BigTorque, label='original torq', color='orange', s=20)
plt.plot(myX, torq_corrected, label='torq estimate (rf corrected)',linewidth=1, alpha=0.6, color='c')
plt.title("measured Torq vs New Torq Est (rf adjusted), " + 'RMSE: ' + str(rmse))
plt.legend()
plt.show()





