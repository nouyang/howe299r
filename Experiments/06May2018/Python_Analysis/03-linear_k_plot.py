"""
Created on 07 May 2018
@author: nrw

Plot original theta, & predicted theta,  vs torque
Plot predicted theta vs original theta 
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shelve
from datetime import datetime

from sklearn import linear_model
from sklearn import metrics

import seaborn as sns

sns.set(rc={'figure.figsize':(20,10)})
#===============================================
#### DECLARE CONSTANTS ####
#===============================================


def plot_fit_and_residuals(infile, infile2, strdatatxt):
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

    torq_estX = torq_est[:,0]
    torq_estY = torq_est[:,1] 

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

    # ---- X position ----

    colorsIdx =  [colorsList[i] for i in BigPosIdx]


    print('~~~~~~~~~~Plotting!' + '~~~~~~~~')
    #===============================================
    #### Plot linear fit ####
    #===============================================




    df=pd.DataFrame({'Resid of TorqX fit':resid[:,0],
                     'Resid of TorqY fit':resid[:,1],
                     'TorqX measured (g*cm)':BigTorque[:,0],
                     'TorqY measured (g*cm)':BigTorque[:,1],
                     'ForceZ (g)':BigForce[:,2],
                     'PositionX (cm)': BigPosition[:,0],
                     'PositionY (cm)': BigPosition[:,1],
                     'ThetaX (deg)': BigTheta[:,0],
                     'ThetaY (deg)': BigTheta[:,1],
                     'ThetaZ (deg)': BigTheta[:,2],
                     'TorqX estimated (g*cm)': torq_est[:,0],
                     'TorqY estimated (g*cm)': torq_est[:,1],
                     'Colors':colorsIdx})

    #===============================================
    #### Plot linear fit ####
    #===============================================

    f, (ax1, ax2) = plt.subplots(1,2)
    ax1.set(xlim=(-550,50), ylim=(-550,50))
    ax2.set(xlim=(-550,50), ylim=(-550,50))
    rg = sns.regplot(data=df, x='TorqX measured (g*cm)', y='TorqX estimated (g*cm)', ax=ax1)
    rg2 = sns.regplot(data=df, x='TorqY measured (g*cm)', y='TorqY estimated (g*cm)', ax=ax2)

    plt.suptitle('Torques X & Y, measured vs estimated\n' + strdatatxt)

    strtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ax = plt.gca()
    plt.text(1.01, 0, 'Time: '+strtime, horizontalalignment='left', verticalalignment='bottom',
            transform = ax.transAxes, fontsize=6)

    #====
    #### Annotate plot with K & RMSE ####
    #===
    linrmse = np.around(metrics.mean_squared_error(BigTorque, torq_est, multioutput='raw_values')**0.5,
                        decimals=2)
    print(K)
    print('Root Mean Squared Error: %s' % str(linrmse))

    plt.text(1.01, 0.2, 'TorqueXYZ linear fit \nRMSE: \n'+ str(linrmse), horizontalalignment='left', verticalalignment='bottom',
            transform = ax.transAxes, fontsize=10)

    plt.text(1.01, 0.4, 'K:\n' + str(np.around(K, decimals=2)), horizontalalignment='left', verticalalignment='bottom',
            transform = ax.transAxes, fontsize=10)

    plt.gcf().savefig('Torq_vs_TorqEst'+strdatatxt)
    # plt.show()


    #===============================================
    #### Plot resids ####
    #===============================================

    # sns.pairplot(data=df, hue='Colors', y_vars=['Resid of TorqX fit', 'Resid of TorqY fit'],
                 # x_vars=[ 'TorqX measured (g*cm)', 'TorqY measured (g*cm)', 
                     # 'TorqX estimated (g*cm)', 'TorqY estimated (g*cm)' ])

    adjust_top = 0.85
    #=== Resid vs Force & Pos 
    sns.pairplot(data=df, hue='Colors', y_vars=['Resid of TorqX fit', 'Resid of TorqY fit'],
                 x_vars=[ 'ForceZ (g)','PositionX (cm)','PositionY (cm)'])
    plt.suptitle('Residuals vs Force & Pos \n' + strdatatxt)
    plt.subplots_adjust(top=adjust_top)
    ax = plt.gca()
    plt.text(1.01, 0, 'Time: '+strtime, horizontalalignment='left', verticalalignment='bottom',
            transform = ax.transAxes, fontsize=6)
    plt.gcf().savefig('Resid_vs_Force_Pos'+strdatatxt)

    #=== Resid vs Theta
    sns.pairplot(data=df, hue='Colors', y_vars=['Resid of TorqX fit', 'Resid of TorqY fit'],
                 x_vars=[ 'ThetaX (deg)' ,'ThetaY (deg)', 'ThetaZ (deg)'])
    plt.suptitle('Residuals vs Thetas\n' + strdatatxt)
    plt.subplots_adjust(top=adjust_top)
    ax = plt.gca()
    plt.text(1.01, 0, 'Time: '+strtime, horizontalalignment='left', verticalalignment='bottom',
            transform = ax.transAxes, fontsize=6)
    plt.gcf().savefig('Resid_vs_Theta'+strdatatxt)

    #=== Resid vs Torq Est 
    sns.pairplot(data=df, hue='Colors', y_vars=['Resid of TorqX fit', 'Resid of TorqY fit'],
                 x_vars=[ 'TorqX estimated (g*cm)', 'TorqY estimated (g*cm)' ])
    plt.suptitle('Residuals vs Torq Estimates \n' + strdatatxt)
    plt.subplots_adjust(top=adjust_top)
    ax = plt.gca()
    plt.text(1.01, 0, 'Time: '+strtime, horizontalalignment='left', verticalalignment='bottom',
            transform = ax.transAxes, fontsize=6)
    plt.gcf().savefig('Resid_vs_Torq'+strdatatxt)

    plt.show()


infile = 'cleaned_data_no_tendon'
infile2 = 'resid_no_tendon'
strdatatxt = 'No tendon load'
plot_fit_and_residuals(infile, infile2, strdatatxt)

infile = 'cleaned_data_loaded_tendon'
infile2 = 'resid_loaded_tendon'
strdatatxt = 'Loaded tendon'
plot_fit_and_residuals(infile, infile2, strdatatxt)
