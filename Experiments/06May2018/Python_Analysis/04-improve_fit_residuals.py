"""
tofix: THIS CODE IS EXTREMELY UGLY AND REPETITIVE 

Calculate residuals and fit with linear & RF models
Collect statistics about new compensated model
Display fit to residuals

todo: Do cross validation

Created May 6 2018
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
from sklearn.preprocessing import PolynomialFeatures

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

myX = BigTheta 
myy = BigTorque[:,1] 
#myY = Residuals 

#======= Adaboost on decision tree
'''
###############################################################
#### Ada Boost Fit of Resid (black box)
###############################################################
ada = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                                                     n_estimators=300, random_state=rng)
ada.fit(myX, myy)
y_est= ada.predict(myX)

rmse = metrics.mean_squared_error(myy, y_est, multioutput='raw_values')**0.5
print(rmse)

plt.scatter(myX[:,1], myy, label='measured torque y', color='orange', s=15)
plt.scatter(myX[:,1], y_est, label='ADA estimated torque y',color='k', s=15, alpha=0.5)
plt.xlabel("theta y (deg)")
plt.ylabel("torque y (g*cm)")
plt.title("AdaBoosted Decision Tree Regression \ndirectly on Torque data (no linear model)\n" 
          + 'RMSE: %.04f' % float(rmse))
plt.legend()
plt.show()

#======= Gradient Boosting Regressor 
###############################################################
#### Gradeint Boost Fit (original data) 
###############################################################
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                    max_depth=1, random_state=0, loss='ls')
gb.fit(myX, myy)
y_est= gb.predict(myX)

rmse = metrics.mean_squared_error(myy, y_est, multioutput='raw_values')**0.5
print(rmse)

plt.plot()
plt.scatter(myX[:,1], myy, label='measured torque y', color='orange', s=15)
plt.scatter(myX[:,1], y_est, label='GB estimated torque y',color='b', s=15, alpha=0.5)
plt.xlabel("theta y (deg)")
plt.ylabel("torque y (g*cm)")
plt.title("Gradient Boost Regression \ndirectly on Torque data (no linear model)\n" 
          + 'RMSE: %.04f' % float(rmse))
plt.legend()
plt.show()


###############################################################
#### RF Fit of Resid (black box)
###############################################################
rf= RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
rf.fit(myX, myy)
y_est= rf.predict(myX) 

rmse = metrics.mean_squared_error(myy, y_est, multioutput='raw_values')**0.5
print(rmse)

plt.scatter(myX[:,1], myy, label='measured torque y', color='orange', s=15)
plt.scatter(myX[:,1], y_est, label='estimated torque y',color='g', s=15, alpha=0.5)
plt.xlabel("theta y (deg)")
plt.ylabel("torque y (g*cm)")
plt.title("Random Forest Regression \ndirectly on Torque data (no linear model)\n" 
          + 'RMSE: %.04f' % float(rmse))
plt.legend()
plt.show()
'''
###############################################################
#### Linear Fit (torq vs theta) 
###############################################################
lin= LinearRegression(fit_intercept=False)
lin.fit(myX, myy)
lin_torq_est= lin.predict(myX) 

rmse = metrics.mean_squared_error(myy, lin_torq_est, multioutput='raw_values')**0.5
print(rmse)

plt.scatter(myX[:,1], myy, label='measured torque y', color='orange', s=15)
plt.scatter(myX[:,1], lin_torq_est, label='estimated torque y', s=15, color='r', alpha=0.5)
plt.xlabel("theta y (deg)")
plt.ylabel("torque y (g*cm)")
plt.title("Linear Regression directly on Torque Data\n" 
          + 'RMSE: %.02f' % float(rmse))
plt.legend()
plt.show()
'''
###############################################################
#### Poly Fit (torq vs theta) 
###############################################################

# z = np.poly1d(np.polyfit(myX,myy,2))

poly = PolynomialFeatures(degree=2) #use polynomial basis function!
X_ = poly.fit_transform(myX)
# predict_ = poly.fit_transform(X)

clf = linear_model.LinearRegression()
clf.fit(X_, myy)
y_est = clf.predict(X_)

rmse = metrics.mean_squared_error(myy, y_est, multioutput='raw_values')**0.5
print(rmse)

plt.scatter(myX[:,1], myy, label='measured torque y', color='orange', s=15)
plt.scatter(myX[:,1], y_est, label='estimated torque y', s=15, color='r', alpha=0.5)
plt.xlabel("theta y (deg)")
plt.ylabel("torque y (g*cm)")
plt.title("Polynomial fit on drectly on Torque data\n" 
          + 'RMSE: %.04f' % float(rmse))
plt.legend()
plt.show()
'''
###############################################################
#### Evaluate linear fit on residuals 
###############################################################


#####################################################################
#### Attempting Linear Correction
#####################################################################

# add LINEAR residuals
myy = resid[:,1] # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Y ONLY 
# print(myy)
linResid = linear_model.LinearRegression() #allow y intercpt
linResid.fit(myX, myy)
linResid_est = linResid.predict(myX) 
# print(lin_y_est)

rmse2 = metrics.mean_squared_error(myy, linResid_est, multioutput='raw_values')**0.5
print('sanity rmse on resids', np.array_str(rmse2, precision=2))

torq_corrected = lin_torq_est + linResid_est 

print('shape!!',torq_corrected.shape)
print('lin2 est reid  shape!!',linResid_est.shape)
print('bigtorq shape', BigTorque.shape)

rmse = metrics.mean_squared_error(BigTorque[:,1], torq_corrected, multioutput='raw_values')**0.5
print('Resid correct Fit with 2nd term (w/ intercept)\n -- rmse: ', np.array_str(rmse, precision=2))

meas_torqY = BigTorque[:,1]

print('myx shape', myX.shape)
print('myY shape', myy.shape)
plt.scatter(myy, meas_torqY, label='resids of linear fit of y', color='orange', s=20)
plt.scatter(linResid_est, meas_torqY, label='lin est of resid', linewidth=1, alpha=0.6, color='b',
            s=20)
plt.title("lin resids fit\n" + 'RMSE:' +np.array_str(rmse, precision=2))
plt.xlabel('theta Y')
plt.ylabel('residY')
plt.legend()
plt.show()

thetaY = myX[:,1]
plt.scatter(thetaY, meas_torqY, label='measured torqY', color='orange', s=20)
plt.scatter(thetaY, torq_corrected, label='torq estimate (lin corrected)', linewidth=1, alpha=0.6,
            color='g', s=20)
plt.xlabel('theta Y (deg)')
plt.ylabel('torqY estimated')
plt.title("measured Torq vs new torq est (lin adjusted)\n" + 'RMSE: ' + np.array_str(rmse, precision=2))
plt.legend()
plt.show()


#####################################################################
#### Attempting Random Forest correction
##################################################################### 
myy = resid[:,1]
rf = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
rf.fit(myX, myy)
rfResid_est= rf.predict(myX)


torq_corrected = lin_torq_est + rfResid_est 

print('shape!!',torq_corrected.shape)
print('lin2 est reid  shape!!',rfResid_est.shape)

meas_torqY = BigTorque[:,1]

print('orig torq', meas_torqY.shape)
print('corrected torque', torq_corrected.shape)

rmse = metrics.mean_squared_error(meas_torqY, torq_corrected, multioutput='raw_values')**0.5
print('Resid correct Fit with 2nd term (w/ intercept)\n -- rmse: ', np.array_str(rmse, precision=2))


print('myx shape', myX.shape)
print('myY shape', myy.shape)
plt.scatter(myy, meas_torqY, label='resids of linear fit of y', color='orange', s=20)
plt.scatter(rfResid_est, meas_torqY, label='rf est of resid', linewidth=1, alpha=0.6, color='b',
            s=20)
plt.title("rf fit on residuals\n" + 'RMSE:' +np.array_str(rmse, precision=2))
plt.xlabel('theta Y')
plt.ylabel('residY')
plt.legend()
plt.show()

thetaY = myX[:,1]
plt.scatter(thetaY, meas_torqY, label='measured torqY', color='orange', s=20)
plt.scatter(thetaY, torq_corrected, label='torq estimate (rf corrected)', linewidth=1, alpha=0.6,
            color='g', s=20)
plt.xlabel('theta Y (deg)')
plt.ylabel('torqY estimated')
plt.title("measured Torq vs new torq est (rf adjusted)\n" + 'RMSE: ' + np.array_str(rmse, precision=2))
plt.legend()
plt.show()

#####################################################################
#### Attempting Adaboost of Gradient Tree correction
##################################################################### 

ada = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                                                     n_estimators=300, random_state=rng)
myy = resid[:,1]
ada.fit(myX, myy)
resid_est= rf.predict(myX)


torq_corrected = lin_torq_est + rfResid_est 

print('shape!!',torq_corrected.shape)
print('lin2 est reid  shape!!',resid_est.shape)

meas_torqY = BigTorque[:,1]

print('orig torq', meas_torqY.shape)
print('corrected torque', torq_corrected.shape)

rmse = metrics.mean_squared_error(meas_torqY, torq_corrected, multioutput='raw_values')**0.5
print('Resid correct Fit with 2nd term (w/ intercept)\n -- rmse: ', np.array_str(rmse, precision=2))


print('myx shape', myX.shape)
print('myY shape', myy.shape)
plt.scatter(myy, meas_torqY, label='resids of linear fit of y', color='orange', s=20)
plt.scatter(resid_est, meas_torqY, label='ada resid est', linewidth=1, alpha=0.6, color='b',
            s=20)
plt.title("Adaboost Decision Tree fit on residuals\n" + 'RMSE:' +np.array_str(rmse, precision=2))
plt.xlabel('theta Y')
plt.ylabel('residY')
plt.legend()
plt.show()

thetaY = myX[:,1]
plt.scatter(thetaY, meas_torqY, label='measured torqY', color='orange', s=20)
plt.scatter(thetaY, torq_corrected, label='torq est (ada corrected)', linewidth=1, alpha=0.6,
            color='g', s=20)
plt.xlabel('theta Y (deg)')
plt.ylabel('torqY estimated')
plt.title("measured Torq vs new torq est\n(adaboost decision tree adjusted) " + 'RMSE: ' + np.array_str(rmse, precision=2))
plt.legend()
plt.show()

#####################################################################
#### Attempting GradientBoost correction
##################################################################### 
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                    max_depth=1, random_state=0, loss='ls')

myy = resid[:,1]
gb = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
gb.fit(myX, myy)
resid_est= gb.predict(myX)


torq_corrected = lin_torq_est + resid_est 

print('shape!!',torq_corrected.shape)
print('lin2 est reid  shape!!',rfResid_est.shape)

meas_torqY = BigTorque[:,1]

print('orig torq', meas_torqY.shape)
print('corrected torque', torq_corrected.shape)

rmse = metrics.mean_squared_error(meas_torqY, torq_corrected, multioutput='raw_values')**0.5
print('Resid correct Fit with 2nd term (w/ intercept)\n -- rmse: ', np.array_str(rmse, precision=2))


print('myx shape', myX.shape)
print('myY shape', myy.shape)
plt.scatter(myy, meas_torqY, label='resids of linear fit of y', color='orange', s=20)
plt.scatter(resid_est, meas_torqY, label='gb est of resid', linewidth=1, alpha=0.6, color='b',
            s=20)
plt.title("gradient boost fit on residuals\n" + 'RMSE:' +np.array_str(rmse, precision=2))
plt.xlabel('theta Y')
plt.ylabel('residY')
plt.legend()
plt.show()

thetaY = myX[:,1]
plt.scatter(thetaY, meas_torqY, label='measured torqY', color='orange', s=20)
plt.scatter(thetaY, torq_corrected, label='torq estimate (gb corrected)', linewidth=1, alpha=0.6,
            color='g', s=20)
plt.xlabel('theta Y (deg)')
plt.ylabel('torqY estimated')
plt.title("measured Torq vs new torq est\n(gradient boost adjusted) " + 'RMSE: ' + np.array_str(rmse, precision=2))
plt.legend()
plt.show()

