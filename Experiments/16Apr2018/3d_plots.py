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
sns.set_context("talk")
# sns.set_palette(sns.color_palette("cubehelix", 8))
sns.set_palette("husl")

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

def pickXColors(yes):
    if yes:
        colorsList = [posX1, posX1, posX1, 
                     posX2, posX2, posX2, 
                     posX3, posX3, posX3, 
                     posX4, posX4, posX4, 
                     posX5, posX5, posX5]

    else:
        colorsList = [posY1, posY1, posY1, posY1, posY1,
                        posY2, posY2, posY2, posY2, posY2,
                        posY3, posY3, posY3, posY3, posY3]
    colorsIdx =  [colorsList[i] for i in BigPosIdx]
    return colorsIdx


# ---- X position ----






def plotResidVs(xvars, filename, colorsXflag, show=False):
    print('~~~~~~~~~~Plotting! Resids vs' + str(xvars) + '~~~~~~~~')
    colorsIdx = pickXColors(colorsXflag)

    df=pd.DataFrame({'Resid of TorqX fit':resid[:,0], 'Resid of TorqY fit':resid[:,1],
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

    # https://xkcd.com/color/rgb/
    # https://seaborn.pydata.org/tutorial/color_palettes.html
    # colors = ["windows blue",  "rust red", "amber", "grey blue", "tangerine", "rust red"]
    if colorsXflag:
        colors = ["rust red", "orange", "sun yellow", "jade green", "sea blue"]
    else:
    # colors = ["robin's egg", "navy blue", "jade green", "rust red", "orange", "sea blue", "orange"]
        colors = ["eggplant purple", "blue", "aqua"] 
    # sns.set_palette(sns.color_palette("RdBu_r", 7))
    sns.set(rc={'lines.markersize':4})#, 'figure.figsize':(10,50)})
    sns.set_palette(sns.xkcd_palette(colors))

    # vs torq
    # sns.pairplot(data=df, hue='Colors', y_vars=['Resid of TorqX fit', 'Resid of TorqY fit'],
             # x_vars=[ 'TorqX measured (g*cm)', 'TorqY measured (g*cm)', 
                 # 'TorqX estimated (g*cm)', 'TorqY estimated (g*cm)' ],
             # kind='reg', plot_kws={'scatter_kws':{'alpha':0.6, 'linewidths':0.2, 'edgecolors':'k'}, 'fit_reg':False})


    # vs force pos
    sns.pairplot(data=df, hue='Colors', y_vars=['Resid of TorqX fit', 'Resid of TorqY fit'],
                     x_vars = xvars,
                     # x_vars=[ 'ForceZ (g)','PositionX (cm)','PositionY (cm)'],
                     kind='reg', plot_kws={'scatter_kws':{'alpha':0.6, 'linewidths':0.2,
                                          'edgecolors':'black'}, 'fit_reg':False}, 
                     size=4, aspect=1)

    # vs theta
    # sns.pairplot(data=df, hue='Colors', y_vars=['Resid of TorqX fit', 'Resid of TorqY fit'],
                 # x_vars=[ 'ThetaX (deg)' ,'ThetaY (deg)', 'ThetaZ (deg)']) 
                 
    plt.suptitle('Residuals Investigation')

    # plot
    sns.set_style('ticks')
    # the size of A4 paper
    strtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ax = plt.gca()
    plt.text(1.1, 0, 'Time: '+strtime, horizontalalignment='left', verticalalignment='bottom',
            transform = ax.transAxes, fontsize=6)
    plt.gcf().savefig(filename)
    # plt.gcf().savefig('TorqResid_Theta')
    if show:
        plt.show()
# print(pickXColors(False))
# plotResidVs([ 'ThetaX (deg)' ,'ThetaY (deg)', 'ThetaZ (deg)'],'resids_theta_coloredY', pickXColors(False))
# plotResidVs([ 'TorqX measured (g*cm)', 'TorqY measured (g*cm)'], 'resids_torq', pickXColors(False))

# vs torq
# sns.pairplot(data=df, hue='Colors', y_vars=['Resid of TorqX fit', 'Resid of TorqY fit'],
         # x_vars= ['TorqX estimated (g*cm)', 'TorqY estimated (g*cm)' ],
         # kind='reg', plot_kws={'scatter_kws':{'alpha':0.6, 'linewidths':0.2, 'edgecolors':'k'}, 'fit_reg':False})

plotResidVs([ 'ThetaX (deg)' ,'ThetaY (deg)', 'ThetaZ (deg)'],'resids_Theta_coloredX', True)
plotResidVs([ 'ThetaX (deg)' ,'ThetaY (deg)', 'ThetaZ (deg)'],'resids_Theta_coloredY', False)
plotResidVs([ 'TorqX measured (g*cm)', 'TorqY measured (g*cm)'], 'resids_Torq_coloredX', True)
plotResidVs([ 'TorqX measured (g*cm)', 'TorqY measured (g*cm)'], 'resids_Torq_coloredX', False)
plotResidVs([ 'ForceZ (g)','PositionX (cm)','PositionY (cm)'], 'resids_Force_coloredX', True)
plotResidVs([ 'ForceZ (g)','PositionX (cm)','PositionY (cm)'], 'resids_Forc_coloredX', False)
# vs force pos
# sns.pairplot(data=df, hue='Colors', y_vars=['Resid of TorqX fit', 'Resid of TorqY fit'],
                 # x_vars - xvars,
                 # # x_vars=
                 # plot_kws={'scatter_kws':{'alpha':0.6, 'linewidths':0.2,
                                      # 'edgecolors':'black'}, 'fit_reg':False}, 
                 # size=3, aspect=1)

# vs theta
# sns.pairplot(data=df, hue='Colors', y_vars=['Resid of TorqX fit', 'Resid of TorqY fit'],
             # x_vars=[ 'ThetaX (deg)' ,'ThetaY (deg)', 'ThetaZ (deg)']) 
             

