"""
Abstract version of 3d_residuals_plot.py, to handle making 14 graphs
Created on Fri Apr 10
@author: nrw
"""

import numpy as np
import matplotlib.pyplot as plt
import shelve
from datetime import datetime

import plotly.offline as po
import plotly.graph_objs as go
from plotly import tools

from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn import metrics

#===============================================
#### DECLARE CONSTANTS ####
#===============================================

with shelve.open('calculated_data', 'r') as shelf:
    BigTheta = shelf['BigTheta']
    BigTorque = shelf['BigTorque']
    BigForce = shelf['BigForce'] 
    BigPosition = shelf['BigPosition'] 

with shelve.open('calculated_data2', 'r') as shelf:
    torq_est = shelf['torq_est'] 
    resid = shelf['resid'] 
    K = shelf['K'] 

torq_estX = torq_est[:,0]
torq_estY = torq_est[:,1] 

def plot_vs_resid(dataA, dataA_list,  dataB = None, dataB_xtitle =''):
    print(dataA.shape)

    y1 = torq_estX
    y2 = torq_estY

    x1 = dataA # e.g. ForceZ
    Atitle, Aunit = dataA_list

    print('~~~~~~~~~~Plotting! residX, residY vs ', Atitle , '~~~~~~~~')


    trace0 = go.Scatter( x = x1, y = y1, mode = 'markers',
        name = 'TorqueX residuals vs ' + Atitle)

    trace1 = go.Scatter( x = x1, y = y2, mode = 'markers', 
        name = 'TorqueY residuals vs ' + Atitle)

    strtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    overall_title='Torque Residuals vs ' + Atitle + \
        '<br>with 3x3 K, using SkLearn LinReg) (IMU data)'## +  \


    layout = go.Layout(
        title = overall_title,
        legend=dict(x=.5, y=0.1) )

    fig = tools.make_subplots(rows=2, cols=1)#, subplot_titles=(trace0.name, trace1.name))

    fig.append_trace(trace0, 1,1)
    fig.append_trace(trace1, 2,1)

    fig['layout'].update(title=overall_title, showlegend=False)

    fig['layout']['xaxis1'].update(title=Atitle + ' ' + Aunit)
    fig['layout']['xaxis2'].update(title=Atitle + ' ' + Aunit + \
        '<br><br>K: ' + np.array_str(K, precision=2) + \
        '<br>Time: ' + strtime )

    fig['layout']['yaxis1'].update(title='TorqueX residual (g*cm)')
    fig['layout']['yaxis2'].update(title='TorqueY residual (g*cm)')

    po.plot(fig)
    return

plot_vs_resid(BigForce[:,2], ['ForceZ', 'g'])
#plot_vs_resid(BigPosition[:,0], ['PositionX', 'cm'])
#plot_vs_resid(BigPosition[:,1], ['PositionY', 'cm'])
#plot_vs_resid(BigTheta[:,0], ['ThetaX', 'deg'])
#plot_vs_resid(BigTheta[:,1], ['ThetaY', 'deg'])
#plot_vs_resid(BigTheta[:,2], ['ThetaZ', 'deg'])
#plot_vs_resid(torq_est[:,0], ['Torq Est X (K*measured thetas)', 'g cm'])
#plot_vs_resid(torq_est[:,1], ['Torq Est Y (K*measured thetas)', 'g cm'])




#===============================================
#### PLOT: Residuals (of Y torque_est - torque) vs Force (Z only)
#===============================================

# xplot = torq_est[:,dim]
# xplot2 = BigForce[:,2]
# yplot = resid[:,dim] 

# trace0 = go.Scatter( x = xplot, y = yplot, mode = 'markers',
    # name = 'resid_torqY vs %s-axis %s estimated'%(names[dim], param))

# trace1 = go.Scatter( x = xplot2, y = yplot, mode = 'markers', 
# name = 'resid_torqY vs Resid vs Z-axis Force, as applied')

# #data = [trace0]

# overall_title='%s-axis %s: Resid vs Force applied (with 3x3 K, using SkLearn LinReg) (IMU data)' % \
    # (names[dim], param) + '<br>K: ' + np.array_str(K, precision=2) + '<br>'

# yaxistitle= 'resid (g cm)'
# xaxistitle= 'force (g)'
# |
# layout = go.Layout(
    # title = overall_title,
    # legend=dict(x=.5, y=0.1) )

# fig = tools.make_subplots(rows=2, cols=1, subplot_titles=(trace0.name, trace1.name))

# fig.append_trace(trace0, 1,1)
# fig.append_trace(trace1, 2,1)

# fig['layout'].update(title=overall_title, showlegend=False)
# fig['layout']['xaxis1'].update(title='%s torque est (g cm)' % (names[dim]))
# fig['layout']['xaxis2'].update(title=xaxistitle)
# fig['layout']['yaxis1'].update(title=yaxistitle)
# fig['layout']['yaxis2'].update(title=yaxistitle)

# #fig = go.Figure(data=data, layout=layout)
