"""
Let's make lots of plots!
Now using seaborn.
Now plotting by the three groups.
Created on Fri Apr 10
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

import seaborn as sns

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

torq_estX = torq_est[:,0]
torq_estY = torq_est[:,1] 

posX1 = 'X pos1 = 4.6 cm'     
posX2 = 'X pos2 = 4.1 cm'     
posX3 = 'X pos3 = 3.5 cm'     
posX4 = 'X pos4 = 3.1 cm'     
posX5 = 'X pos5 = 2.6 cm'     

posY1 = 'Y pos1 = 0.4 cm'
posY2 = 'Y pos2 = 0.1 cm'
posY3 = 'Y pos3 = -0.2 cm' 


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


# ----------------------- Plot pair (or more) of plots
# sns.pairplot(data=df, hue='Colors', y_vars=['Resid of TorqX fit', 'Resid of TorqY fit'],
             # x_vars=[ 'TorqX measured (g*cm)', 'TorqY measured (g*cm)', 
                 # 'TorqX estimated (g*cm)', 'TorqY estimated (g*cm)' ])

# sns.pairplot(data=df, hue='Colors', y_vars=['Resid of TorqX fit', 'Resid of TorqY fit'],
             # x_vars=[ 'ForceZ (g)','PositionX (cm)','PositionY (cm)'])


sns.pairplot(data=df, hue='Colors', y_vars=['Resid of TorqX fit', 'Resid of TorqY fit'],
             x_vars=[ 'ThetaX (deg)' ,'ThetaY (deg)', 'ThetaZ (deg)'])


                 # 'TorqX measured (g*cm)'
                 # 'TorqY measured (g*cm)'
                 # 'ForceZ (g)'
                 # 'PositionX (cm)'
                 # 'PositionY (cm)'
                 # 'ThetaX (deg)'
                 # 'ThetaY (deg)'
                 # 'ThetaZ (deg)'
                 # 'TorqX estimated (g*cm)'
                 # 'TorqY estimated (g*cm)'
plt.suptitle('Residual investigation')
strtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
ax = plt.gca()
plt.text(1.1, 0, 'Time: '+strtime, horizontalalignment='left', verticalalignment='bottom',
        transform = ax.transAxes, fontsize=6)

plt.show()

#sns.distplot(df['stand_square_feet'],kde = False, ax=ax[1][2])


