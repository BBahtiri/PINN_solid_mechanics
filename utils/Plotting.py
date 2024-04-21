#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from matplotlib.ticker import FuncFormatter

def plot_pts(Xint, Xbnd):
    '''
    Plots the collcation points from the interior and boundary

    Parameters
    ----------
    Xint : TYPE
        DESCRIPTION.
    Xbnd : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
     #plot the boundary and interior points
    plt.scatter(Xint[:,0], Xint[:,1], s=0.5)
    plt.scatter(Xbnd[:,0], Xbnd[:,1], s=1, c='red')
    plt.title("Boundary and interior collocation points")
    plt.show()
    
    
    
def plot_field_2d(Xpts,field, numPtsU, numPtsV, figHeight, figWidth, title=""):
    '''
    '''
    minimum = min(field)
    maximum = max(field)
    plt.figure(figsize =(figWidth,figHeight))
    plt.scatter(Xpts[:,0], Xpts[:,1], s = 5, c = field, cmap = 'jet', vmin = minimum, 
                vmax = maximum)
    plt.axis('equal')
    fmt = lambda x, pos: '{:.10}'.format(x)
    plt.colorbar(format=FuncFormatter(fmt))
    plt.title(title)
    plt.savefig(title + ".pdf", dpi=700, facecolor='w', edgecolor='w', 
                    transparent = 'true', bbox_inches = 'tight')
    plt.show()
    plt.close()
    # xPtsPlt = np.resize(Xpts[:,0], [numPtsV, numPtsU])
    # yPtsPlt = np.resize(Xpts[:,1], [numPtsV, numPtsU])
    # fieldPtsPlt = np.resize(Ypts, [numPtsV, numPtsU])
    # plt.contourf(xPtsPlt, yPtsPlt, fieldPtsPlt, 255, cmap=plt.cm.jet)
    # plt.colorbar()
    # plt.title(title)
    # plt.show()

def plotForceDisp(fdGraph,figHeight,figWidth):           
    
    filename = "Force-Displacement"
    plt.figure(figsize=(figWidth, figHeight))
    plt.plot(fdGraph[:,0], fdGraph[:,1])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Displacement',fontweight='bold',fontsize=14)
    plt.ylabel('Force',fontweight='bold',fontsize=14)
    plt.tight_layout()
    plt.savefig(filename + ".pdf", dpi=700, facecolor='w', edgecolor='w', 
                    transparent = 'true', bbox_inches = 'tight')
    plt.show()
    plt.close()
    
def createFolder(folder_name):
    try:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
    except OSError:
        print ('Error: Creating folder. ' +  folder_name)    
