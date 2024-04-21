# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 11:29:50 2023

@author: bahtiri
"""
import numpy as np
import tensorflow as tf
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from fnmatch import fnmatch
from tensorflow.keras import layers
from tensorflow import keras
from scipy.ndimage import interpolation
import scipy.io


def resize_array(x):
    i = 1000 # Sequence length
    z = i / len(x) 
    x_res = np.expand_dims(interpolation.zoom(x[:,0],z),axis=-1)    
    return x_res

# Import data from mat file
def getData_exp(inputF):
    C = scipy.io.loadmat(inputF)
    stress = C['expStress'].astype('float64')
    timeVec = C['timeVec'].astype('float64')
    trueStrain = (C['trueStrain'].astype('float64')+1)
    # Resize the arrays by using interpolation to reduce sequence length 
    stress_res = resize_array(stress)
    timeVec_res = resize_array(timeVec)
    trueStrain_res = resize_array(trueStrain)
    
    
    deltaT = np.mean(np.diff(timeVec,axis=0))
    deltaT_res = np.mean(np.diff(timeVec_res,axis=0))
    if deltaT_res < 0:
        stress_res = stress_res[:-10]
        timeVec_res = timeVec_res[:-10]
        trueStrain_res = trueStrain_res[:-10]
        deltaT_res = np.mean(np.diff(timeVec_res,axis=0))
        
    return stress_res, trueStrain_res, deltaT_res, timeVec_res

# Import data from mat file
def getData(inputF):
    C = scipy.io.loadmat(inputF)
    s = C['t'].astype('float64')
    x = C['x'].astype('float64')
    
    return s, x

def Dev(A):
    """Deviatoric part of a tensor"""
    I = tf.eye(2)
    return A - (1 / 2) * tf.linalg.trace(A) * I


