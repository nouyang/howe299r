"""
Created on 07 May 2018
@author: nrw

Calculates linear fit and plots torque estimate
This is the "physical model" fit part, where we asssume a linear fit is appropriate
Adds linear fit torque estimate XYZ, linear fit residuals XYZ to shelved data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn import metrics
import shelve

import seaborn as sns
sns.set()

def calc_resid(infname, csvout, outfname):
    with shelve.open(infname, 'r') as shelf:
        BigTheta = shelf['BigTheta']
        BigTorque = shelf['BigTorque']
        BigForce = shelf['BigForce'] 
        BigPosition = shelf['BigPosition'] 
        BigPosIdx = shelf['BigPosIdx']

    #===============================================
    #### DECLARE CONSTANTS ####
    #===============================================

    print('number of datapoints', BigTheta.shape)
    resid = []
    torq_est = []

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


    print('\n==========')
    print('LinReg K Coefficients: \n', K_lin)
    print('\n====')

    #######################################
    # --  torq_est Definition 
    #######################################
    torq_est = yPredLin

    # --  quick sanity check 
    resid = torq - torq_est 
    print('Residual mean: ', resid.mean(axis=0))
    print('Residual Std Dev: ', resid.std(axis=0))
    # print('Linear fit torq est mean: ', torq_est.mean(axis=0))
    # print('Linear fit torq Est Std Dev: ', torq_est.std(axis=0))

    #print('Variance score (ideal 1): %.2f' % r2_score(thetaY))
    #print('Mean Absolute Error: %0.02f' % metrics.mean_absolute_error(torq, yPred))  

    #######################################
    # -- Fit metrics 
    #######################################
    print('\n===SkLearn Metrics===')

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

    f=open(csvout, 'w')
    f.write('Big Pos, , , Big Force, , , BigTheta, , , BigTorque, , ,  \
        Torq Est, , , Resid, , BigPosIdx\n')
    f.write('X, Y, Z,'*6 + ' \n')
    f.close()
    f=open(csvout, 'a')
    np.savetxt(f, full_data, delimiter=",", fmt='%0.02f')

    with shelve.open(outfname, 'c') as shelf:
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
                     'Theta Y (deg)': BigTheta[:,1]
                     })

calc_resid('cleaned_data_no_tendon', 'imu_data_no_tendon', 'resid_no_tendon')
calc_resid('cleaned_data_loaded_tendon', 'imu_data_loaded_tendon', 'resid_loaded_tendon')
