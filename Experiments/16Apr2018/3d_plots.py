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

# color_map = ['grey', 'grey', 'grey', 
             # 'orange', 'orange', 'orange',
             # 'navy', 'navy', 'navy',
             # 'green', 'green', 'green', 
             # 'slateblue', 'slateblue', 'slateblue']

# color_map = {0:'grey', 1:'grey', 2:'grey', 
             # 3:'orange', 4:'orange', 5:'orange',
             # 6:'navy', 7:'navy', 8:'navy',
             # 9:'green', 10:'green', 11:'green', 
             # 12:'slateblue', 13:'slateblue', 14:'slateblue'}

# colorsList = [10, 10, 10,   
              # 20, 20, 20, 
              # 30, 30, 30, 
              # 40, 40, 40, 
              # 50, 50, 50]
posX1 = 'X pos1 = 4.6 cm'     
posX2 = 'X pos2 = 4.1 cm'     
posX3 = 'X pos3 = 3.5 cm'     
posX4 = 'X pos4 = 3.1 cm'     
posX5 = 'X pos5 = 2.6 cm'     

posY1 = 'Y pos1 = 0.4 cm'
posY2 = 'Y pos2 = 0.1 cm'
posY3 = 'Y pos3 = -0.2 cm' 

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

residX = resid[:,0]
residY = resid[:,1]
df=pd.DataFrame({'Torq X measured':  BigTorque[:,0], 
                 'Torq Y measured':  BigTorque[:,1], 
                 'Torq X estimated':  torq_est[:,0],
                 'Torq Y estimated':  torq_est[:,1],
                 'Colors':colorsIdx})

sns.set()
g = sns.lmplot(x='Torq X measured', y='Torq X estimated', data=df, hue='Colors')
plt.title('95% confidence interval')

plt.show()

def plot_vs_resid(vsResid, vsResid_text):

    # y1 = torq_estX[0:50]
    # y2 = torq_estY[0:50]
    # response = response[0:50]
    # posIdx = posIdx[0:50]
    # colorsIdx=colorsIdx[0:50]

    residX = resid[:,0]
    residY = resid[:,1]

    posIdx = BigPosIdx
    # DUH. 
    ytitle = 'Residuals'
    yunits = '(g*cm)'

    x = vsResid # e.g. ForceZ
    horiztitle, horizunits = vsResid_text 
    # aDF= pd.DataFrame(response, columns= Atitle)
    # aDF['ColorIdx'] = colorsIdx

    print('~~~~~~~~~~Plotting! ' + ytitle + ' of torq X Y fits, vs ', horiztitle,
          '~~~~~~~~')

    df=pd.DataFrame({'Resid of TorqX fit':residX, 
                     'Resid of TorqY fit':residY, horiztitle:vsResid, 'Colors':colorsIdx})

    # g = sns.PairGrid(df, hue='posIdx') 
    # hue=posIdx -> setting an array element with a sequence, b/c var, not text
    g = sns.lmplot(x=horiztitle, y='Resid of TorqY fit', data=df, hue='Colors', size=4, aspect=1)
    g.add_legend()
    plt.show()

#sns.distplot(df['stand_square_feet'],kde = False, ax=ax[1][2])


#plot_vs_resid(BigTorque[:,0],  ['Torque X Real (Measured)', 'g*cm'])
#plot_vs_resid(BigTorque[:,1], ['Torque Y Real (Measured)', 'g*cm'])
# plot_vs_resid(BigForce[:,2], ['ForceZ', 'g'])
# plot_vs_resid(BigPosition[:,0], ['PositionX', 'cm'])
# plot_vs_resid(BigPosition[:,1], ['PositionY', 'cm'])
# plot_vs_resid(BigTheta[:,0], ['ThetaX', 'deg'])
# plot_vs_resid(BigTheta[:,1], ['ThetaY', 'deg'])
# plot_vs_resid(BigTheta[:,2], ['ThetaZ', 'deg'])
# plot_vs_resid(torq_est[:,0], ['Torq Est X (K*measured thetas)', 'g cm'])
# plot_vs_resid(torq_est[:,1], ['Torq Est Y (K*measured thetas)', 'g cm'])
