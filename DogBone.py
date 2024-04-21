#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D linear elasticity example
"""
import sys
import tensorflow as tf
import numpy as np
import time
import os
import scipy.optimize
from utils.scipy_loss import scipy_function_factory
from utils.Solvers import Hyperelasticity2D_coll_dist
from utils.Plotting import plot_pts
from utils.Plotting import plot_field_2d
from utils.Plotting import plotForceDisp
from utils.Plotting import createFolder
from utils.misc import Tri_Segm
#make figures bigger on HiDPI monitors
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 200
np.random.seed(42)
tf.random.set_seed(42)
import scipy.io
import tracemalloc
from math import pi, cos, sin, sqrt
from random import random
from matplotlib import path
import heapq
import itertools
from scipy.ndimage import interpolation

class DogBone(Hyperelasticity2D_coll_dist):
    '''
    Class including the boundary conditions and NN
    '''       
    def __init__(self, layers, train_op, num_epoch, print_epoch, model_data, data_type,length_phys,length_phys_bnd):        
        super().__init__(layers, train_op, num_epoch, print_epoch, model_data, data_type,length_phys,length_phys_bnd)
       
    #@tf.function
    def dirichletBound(self, X, xPhys, yPhys,u_delta):    
        # multiply by x,y for strong imposition of boundary conditions
        u_val = X[:,0:1]
        v_val = X[:,1:2]
        
        # u_val = (1-xPhys) * xPhys*u_val + u_delta*xPhys
        #u_val = xPhys*(54-xPhys)*u_val + u_delta*xPhys/54
        #u_val = xPhys * u_val + u_delta*xPhys/54
        u_val = xPhys*u_val
        v_val = yPhys*v_val
        
        return u_val, v_val


def getF_D(inputF):
    C = scipy.io.loadmat(inputF)
    D = C['Displacement'].astype('float64')
    F = C['Force'].astype('float64')
    T = C['timeVec'].astype('float64')
    
    D_final = resize_array(D)
    F_final = resize_array(F)
    T_final = resize_array(T)
    deltaT = np.mean(np.diff(T_final,axis=0))
    
    return D_final,F_final,T_final
    
def getUnifIntPts(inputF):
    C = scipy.io.loadmat(inputF)
    xPhys_int = C['X'].astype('float64')
    yPhys_int = C['Y'].astype('float64')
    
    return xPhys_int, yPhys_int


def getUnifEdgePts(inputF,face):
    C = scipy.io.loadmat(inputF)
    
    if face==1:
        xPhys = C['xTopArc'].astype('float64')
        yPhys = C['yTopArc'].astype('float64')
        xNorm = C['xNormA'].astype('float64')
        yNorm = C['yNormA'].astype('float64')
    elif face==2:
        xPhys = C['xTopLine'].astype('float64')
        yPhys = C['yTopLine'].astype('float64')
        xNorm = np.zeros_like(xPhys).astype('float64')
        yNorm = np.ones_like(yPhys).astype('float64')
    elif face==3:
        xPhys = C['xLeftArc'].astype('float64')
        yPhys = C['yLeftArc'].astype('float64')
        xNorm = C['xNormC'].astype('float64')
        yNorm = C['yNormC'].astype('float64')
    elif face==4:
        xPhys = C['xLeftLine'].astype('float64')
        yPhys = C['yLeftLine'].astype('float64')
        xNorm = np.ones_like(xPhys).astype('float64')*(-1)
        yNorm = np.zeros_like(yPhys).astype('float64')
    elif face==5:
        xPhys = C['xBotLine'].astype('float64')
        yPhys = C['yBotLine'].astype('float64')
        xNorm = np.zeros_like(xPhys).astype('float64')
        yNorm = np.ones_like(yPhys).astype('float64')*(-1)
    elif face==6:
        xPhys = C['xRightLine'].astype('float64')
        yPhys = C['yRightLine'].astype('float64')
        xNorm = np.ones_like(xPhys).astype('float64')
        yNorm = np.zeros_like(yPhys).astype('float64')
    
    return xPhys, yPhys, xNorm, yNorm


def add_points(indx,indy,Xint,Yint,r,Xbnd_all,state,init_length):
    ind = np.concatenate((indx,indy),axis=0)
    Xbnd_crd = Xbnd_all[:,0]
    Ybnd_crd = Xbnd_all[:,1]
    state_updated = state
    for i in range(len(ind)):
        current_crd = Xint[ind[i]]
        current_state = state[:,ind [i]]
        cs = current_state[:, :, np.newaxis]
        points = ReplicateNTimes(UniformRandomPointInCircle,r,current_crd[0],current_crd[1]
                        ,Xbnd_crd,Ybnd_crd,Ntrials=50)
        pts_len = len(points)
        state_toadd = np.repeat(cs,pts_len,axis=1)
        Xint = np.concatenate((Xint,points))
        state_updated = np.concatenate([state_updated,state_toadd],axis=1)
        
        
        
    Yint = np.zeros_like(Xint).astype(data_type) 
    state_final = tf.convert_to_tensor(state_updated)
    
    return Xint, Yint, state_final
    
def ReplicateNTimes(func, rad, xc, yc,Xbnd,Ybnd,Ntrials=50):
    xpoints, ypoints = [], []
    for _ in range(Ntrials):
        xp,yp = func(rad, xc, yc,Xbnd,Ybnd)
        xpoints.append(xp)
        ypoints.append(yp)
    inpoints = inpolygon(np.array(xpoints),np.array(ypoints),Xbnd,Ybnd)   
    to_keep = np.where(inpoints)[0]
    point = np.vstack((xpoints,ypoints)).T
    point = point[to_keep]
    return point
    
def UniformRandomPointInCircle(inputRadius,xcenter,ycenter,Xbnd,Ybnd):   
    r = inputRadius*sqrt(random())
    theta = 2 * pi * random()
    x = xcenter + r * cos(theta)
    y = ycenter + r * sin(theta)
    return x,y


def inpolygon(xq, yq, xv, yv):
    shape = xq.shape
    xq = xq.reshape(-1)
    yq = yq.reshape(-1)
    xv = xv.reshape(-1)
    yv = yv.reshape(-1)
    q = [(xq[i], yq[i]) for i in range(xq.shape[0])]
    p = path.Path([(xv[i], yv[i]) for i in range(xv.shape[0])])
    return p.contains_points(q).reshape(shape)    

def resize_array(x):
    i = 500 # Sequence length
    z = i / len(x) 
    x_res = np.expand_dims(interpolation.zoom(x[:,0],z),axis=-1)    
    return x_res

    
model_data = dict()
model_data["Temperature"] = 296
model_data["zita"] = 0.0
model_data["wgf"] = 0.0 
model_data["wnp"] = 0.0
model_data["state"] = "plane strain"
model_data["E"] = 760
model_data["nu"] = 0.23


file_coll = 'DogBone6.mat'
xPhys, yPhys = getUnifIntPts(file_coll)
dis,force,time_sim = getF_D('epoxy_dry_rt')
deltaT = np.diff(time_sim,axis=0)
model_data["dt"] = deltaT
deltaU = np.diff(dis,axis=0)
data_type = "float64"
numPtsU = len(xPhys)
numPtsV = len(yPhys)

Xint = np.concatenate((xPhys,yPhys),axis=1).astype(data_type)
Yint = np.zeros_like(Xint).astype(data_type)
# prepare boundary points in the fromat Xbnd = [Xcoord, Ycoord, dir] and
# Ybnd = [trac], where Xcoord, Ycoord are the x and y coordinate of the point,
# dir=0 for the x-component of the traction and dir=1 for the y-component of 
# the traction. xNorm and yNorm are the normal vectors of the boundaries

#first arc -> face  = 1
xPhysBndA, yPhysBndA , xNormA, yNormA = getUnifEdgePts(file_coll,1)
dirA0 = np.zeros_like(xPhysBndA) #x-direction    
dirA1 = np.ones_like(xPhysBndA)  #y-direction    
XbndA0 = np.concatenate((xPhysBndA, yPhysBndA, xNormA, yNormA, dirA0), axis=1).astype(data_type)
XbndA1 = np.concatenate((xPhysBndA, yPhysBndA, xNormA, yNormA, dirA1), axis=1).astype(data_type)
    
#top boundary line -> face = 2
xPhysBndB, yPhysBndB , xNormB, yNormB = getUnifEdgePts(file_coll,2)
dirB0 = np.zeros_like(xPhysBndB) #x-direction    
dirB1 = np.ones_like(xPhysBndB)  #y-direction    
XbndB0 = np.concatenate((xPhysBndB, yPhysBndB, xNormB, yNormB, dirB0), axis=1).astype(data_type)
XbndB1 = np.concatenate((xPhysBndB, yPhysBndB, xNormB, yNormB, dirB1), axis=1).astype(data_type)

#second arc -> face = 3
# xPhysBndC, yPhysBndC , xNormC, yNormC = getUnifEdgePts(file_coll,3)
# dirC0 = np.zeros_like(xPhysBndC) #x-direction    
# dirC1 = np.ones_like(xPhysBndC)  #y-direction    
# XbndC0 = np.concatenate((xPhysBndC, yPhysBndC, xNormC, yNormC, dirC0), axis=1).astype(data_type)
# XbndC1 = np.concatenate((xPhysBndC, yPhysBndC, xNormC, yNormC, dirC1), axis=1).astype(data_type)


#left boundary -> face = 4
xPhysBndD, yPhysBndD, xNormD, yNormD = getUnifEdgePts(file_coll,4)
dirD0 = np.zeros_like(xPhysBndD) #x-direction    
dirD1 = np.ones_like(xPhysBndD)  #y-direction    
XbndD0 = np.concatenate((xPhysBndD, yPhysBndD, xNormD, yNormD, dirD0), axis=1).astype(data_type)
XbndD1 = np.concatenate((xPhysBndD, yPhysBndD, xNormD, yNormD, dirD1), axis=1).astype(data_type)

#bottom boundary -> face = 5
xPhysBndE, yPhysBndE, xNormE, yNormE = getUnifEdgePts(file_coll,5)
dirE0 = np.zeros_like(xPhysBndE) #x-direction    
dirE1 = np.ones_like(xPhysBndE)  #y-direction    
XbndE0 = np.concatenate((xPhysBndE, yPhysBndE, xNormE, yNormE, dirE0), axis=1).astype(data_type)
XbndE1 = np.concatenate((xPhysBndE, yPhysBndE, xNormE, yNormE, dirE1), axis=1).astype(data_type)

#right boundary -> face = 6
xPhysBndF, yPhysBndF, xNormF, yNormF = getUnifEdgePts(file_coll,6)
dirF0 = np.zeros_like(xPhysBndF) #x-direction    
dirF1 = np.ones_like(xPhysBndF)  #y-direction    
XbndF0 = np.concatenate((xPhysBndF, yPhysBndF, xNormF, yNormF, dirF0), axis=1).astype(data_type)
XbndF1 = np.concatenate((xPhysBndF, yPhysBndF, xNormF, yNormF, dirF1), axis=1).astype(data_type)

plt.quiver(xPhysBndA,yPhysBndA,xNormA,yNormA)
plt.quiver(xPhysBndB,yPhysBndB,xNormB,yNormB)
plt.quiver(xPhysBndD,yPhysBndD,xNormD,yNormD)
plt.quiver(xPhysBndE,yPhysBndE,xNormE,yNormE)
plt.quiver(xPhysBndF,yPhysBndF,xNormF,yNormF)
plt.show()

#define loading (traction free faces)
YbndA0 = np.zeros_like(xPhysBndA).astype(data_type)
YbndA1 = np.zeros_like(yPhysBndA).astype(data_type)
YbndB0 = np.zeros_like(xPhysBndB).astype(data_type)
YbndB1 = np.zeros_like(yPhysBndB).astype(data_type)
YbndD1 = np.zeros_like(yPhysBndD).astype(data_type)
YbndD0 = np.zeros_like(xPhysBndD).astype(data_type)
YbndE0 = np.zeros_like(xPhysBndE).astype(data_type)
YbndE1 = np.zeros_like(yPhysBndE).astype(data_type)
YbndF1 = np.zeros_like(yPhysBndF).astype(data_type)
YbndF0 = np.zeros_like(xPhysBndF).astype(data_type)

# concatenate all the boundaries
# Xbnd = np.concatenate((XbndD1, XbndE0,XbndF0, XbndF1), axis=0)    
# Xbnd = np.concatenate((XbndD1,XbndA0,XbndB0,XbndC0,
#                         XbndE0, XbndF1), axis=0) 
# Xbnd = np.concatenate((XbndD0,
#                         XbndE1), axis=0) 

Xbnd = np.concatenate((XbndB0,XbndB1,XbndD1,XbndE0,XbndF1),axis=0)
XbndArc = np.concatenate((XbndA0,XbndA1),axis=0)
XbndDis = XbndF0
Xbnd_all = np.concatenate((XbndA0,XbndB0,XbndD0,XbndE0,XbndF0),axis=0)
# concatenate all the loadings
# Ybnd = np.concatenate((YbndD1, YbndE0,YbndF0, YbndF1), axis=0)  
# Ybnd = np.concatenate((YbndD1,YbndA0,YbndB0,YbndC0,
#                         YbndE0, YbndF1), axis=0)  
# Ybnd = np.concatenate((YbndD0, 
#                         YbndE1), axis=0)  

Ybnd = np.concatenate((YbndB0,YbndB1,YbndD1,YbndE0,YbndF1),axis=0)
YbndArc = np.concatenate((YbndA0,YbndA1),axis=0)
YbndDis = YbndF0

#plot the collocation points
plot_pts(Xint, Xbnd[:,0:2])
C = scipy.io.loadmat(file_coll,squeeze_me=True)
model_data['ar'] = C['A'].astype('float64')
model_data['l_arc'] = C['l_arc']
model_data['l_other'] = C['l_others']
# xy =  np.concatenate((xPhys,yPhys), axis=1)
# model_data['ns'] = tf.constant(len(xPhys),dtype=tf.float64)
# model_data['dx'] = tf.constant(0.2,dtype=tf.float64)
# ar = Tri_Segm(xy, len(xPhys))
# model_data['ar'] = tf.constant(ar,dtype=tf.float64)
#model_data['h'] = tf.constant(10,dtype=tf.float64)
#model_data['xi_1'] = tf.constant(model_data['h']/model_data["E"]**2,dtype=tf.float64)
#model_data['xi_2'] = tf.constant(1/model_data["E"]**2,dtype=tf.float64)


#define the model 
tf.keras.backend.set_floatx(data_type)
initializier = tf.keras.initializers.GlorotNormal(seed=1)
initializier2 = tf.keras.initializers.GlorotNormal(seed=10)
initializier3 = tf.keras.initializers.GlorotNormal(seed=100)
initializier4 = tf.keras.initializers.GlorotNormal(seed=1000)
initializier5 = tf.keras.initializers.GlorotNormal(seed=123)
hidden_size = 30
l1 = tf.keras.layers.Dense(hidden_size, "swish",kernel_initializer=initializier)
l2 = tf.keras.layers.Dense(hidden_size, "swish",kernel_initializer=initializier2)
l3 = tf.keras.layers.Dense(hidden_size, "swish",kernel_initializer=initializier3)
l4 = tf.keras.layers.Dense(hidden_size, "swish",kernel_initializer=initializier4)
l5 =  tf.keras.layers.Dense(hidden_size, "swish",kernel_initializer=initializier5) 
l7 = tf.keras.layers.Dense(2, None)
train_op = tf.keras.optimizers.Adam(learning_rate=1e-3,clipnorm=1.0)
num_epoch = 1000
num_epoch1 = 20
print_epoch1 = 1
print_epoch = 10
length_phys = len(Xint)
length_phys_bnd = len(Xbnd)
length_phys_arc = len(XbndArc)
length_phys_disp = len(xPhysBndF)
face_length= len(XbndF0)
figHeight = 5
figWidth = 5
originalDir = os.getcwd()
foldername = 'DogBone_results_hyperelasticity_'+str(hidden_size)    
createFolder('./'+ foldername + '/')
os.chdir(os.path.join(originalDir, './'+ foldername + '/'))
pred_model = DogBone([l1, l2, l3,l4,l5,l7], train_op, num_epoch, 
                                    print_epoch, model_data, data_type,length_phys,length_phys_bnd)

nSteps = len(time_sim) # Total number of steps
fdGraph = np.zeros((nSteps,2),dtype = np.float64)
N_b = len(XbndF0)# Number of boundary points on face D

#convert the training data to tensors
Ybnd_tf = tf.convert_to_tensor(Ybnd)
YbndArc_tf = tf.convert_to_tensor(YbndArc)
YbndDis_tf = tf.convert_to_tensor(YbndDis)
train_bfgs = 1
train_bfgs_vevp = 1
Yint_tf = tf.convert_to_tensor(Yint)


#pred_model.built= True
#pred_model.load_weights('/bigwork/nhgebaht/ML_Python/PINN_Betim_DogBone_Adaptive_22/DogBone_results_adaptive_100/model_weights_1')
for iStep in range(0,nSteps):
    Xint_tf = tf.convert_to_tensor(Xint)
    Xbnd_tf = tf.convert_to_tensor(Xbnd)
    XbndArc_tf = tf.convert_to_tensor(XbndArc)
    XbndDis_tf = tf.convert_to_tensor(XbndDis)
    U_curr = dis[iStep]
    deltaU_curr = deltaU[iStep]
    dt = deltaT[iStep]
    print("Number of collocation points: ", len(Xint),file=sys.stderr)
    print("Current Step: ", iStep)
    print("Current Displacement Increment: ", deltaU_curr)
    print("Current Displacement: ", U_curr)
    u_delta = tf.constant(deltaU_curr ,dtype=tf.float64)
    YbndDis_tf = tf.zeros((length_phys_disp,1),dtype = np.float64) + deltaU_curr
    plot_pts(Xint_tf, Xbnd[:,0:2])
    print("Delta u:", deltaU_curr ,file=sys.stderr)   
    # #Train with VeVp model
    t0 = time.time()
    if iStep ==0:
        print("Training (ADAM)...",file=sys.stderr)
        pred_model.network_learn(Xint_tf, Yint_tf, Xbnd_tf, Ybnd_tf,u_delta,XbndArc_tf,YbndArc_tf,XbndDis_tf,
                                       YbndDis_tf)
        t1 = time.time()
        print("Time taken (ADAM)", t1-t0, "seconds",file=sys.stderr)
        
    print("Training (BFGS)...",file=sys.stderr)
    loss_func = scipy_function_factory(pred_model, Xint_tf, Yint_tf, Xbnd_tf, Ybnd_tf,u_delta,XbndArc_tf,YbndArc_tf,
                                       XbndDis_tf,YbndDis_tf)
    # convert initial model parameters to a 1D tf.Tensor
    init_params = np.float64(tf.dynamic_stitch(loss_func.idx, pred_model.trainable_variables).numpy())
    results = scipy.optimize.minimize(fun=loss_func, x0=init_params,jac=True,
                                          method="BFGS", options={'disp': 1, 
            'gtol': 1e-8, 'eps': 1e-8, 'maxiter': 200})  
    # after training, the final optimized parameters are still in results.position
    # so we have to manually put them back to the model
    loss_func.assign_new_model_parameters(results.x)   
    t2 = time.time()
    print("Time taken (all)", t2-t0, "seconds",file=sys.stderr)
    print("Training finished!")     
    ##############################################################################
    #Update state variables
    #print("Update interior state!")
    #state = pred_model.UpdateState(Xint_tf[:,0:1], Xint_tf[:,1:2]  , u_delta, state)
    #print("Update boundary state!")
    #stateB= pred_model.UpdateState(Xbnd_tf[:,0:1], Xbnd_tf[:,1:2], u_delta, stateB)
    #print("Update arc boundary state!")
    #stateC= pred_model.UpdateState(XbndArc_tf[:,0:1], XbndArc_tf[:,1:2], u_delta, stateC)
    #print("Update displacement boundary state!")
    #stateD= pred_model.UpdateState(XbndDis_tf[:,0:1], XbndDis_tf[:,1:2], u_delta, stateD)
    #############################################################################
    #Use the trained NN to predict now
    print("Testing...",file=sys.stderr)
    XTest_tf = Xint_tf
    xPhysTest = XTest_tf[:,0]
    yPhysTest = XTest_tf[:,1]
    xPhysTest = xPhysTest[:,tf.newaxis]
    yPhysTest = yPhysTest[:,tf.newaxis]
    print("Coordinates...",file=sys.stderr)
    YTest = pred_model(XTest_tf,u_delta).numpy()
    YTest_bnd = pred_model(Xbnd_tf,u_delta).numpy()
    YTest_bnd_arc = pred_model(XbndArc_tf,u_delta).numpy()
    YTest_bnd_dis = pred_model(XbndDis_tf,u_delta).numpy()
    print("Calculate tractions...",file=sys.stderr)
    traction_predx, traction_predy = pred_model.tractions(XbndDis_tf,u_delta)
    fdGraph[iStep,0] = u_delta
    fdGraph[iStep,1] = np.sum(traction_predx[:])/N_b  
    #Stress
    print("Calculate stress...",file=sys.stderr)
    stress_xx_comp, stress_yy_comp, stress_xy_comp = pred_model.constitutiveEq(xPhysTest, yPhysTest,u_delta)
    stress_xx_comp = stress_xx_comp.numpy()
    stress_yy_comp = stress_yy_comp.numpy()
    stress_xy_comp = stress_xy_comp.numpy()
    #Strain
    print("Calculate strain...",file=sys.stderr)
    eps_xx_comp, eps_yy_comp, eps_xy_comp = pred_model.green_strain(xPhysTest, yPhysTest,u_delta)
    eps_xx_comp = eps_xx_comp.numpy()
    eps_yy_comp = eps_yy_comp.numpy()
    eps_xy_comp = eps_xy_comp.numpy() 
    print("'Save as .mat'...",file=sys.stderr)
    scipy.io.savemat('out_{}.mat'.format(iStep),{'x': xPhys, 'y': yPhys, 
                                         'u': YTest, 's11': stress_xx_comp,
                                         's22': stress_yy_comp, 's12': stress_xy_comp,
                                         'E_xx': eps_xx_comp,
                                         'E_yy': eps_yy_comp,
                                         'E_xy': eps_xy_comp})

    print("Calculate stress at boundary...",file=sys.stderr)
    stress_xx_dis, stress_yy_dis, stress_xy_dis = pred_model.constitutiveEq(XbndDis_tf[:,0:1], XbndDis_tf[:,1:2],u_delta)
    stress_xx_dis = stress_xx_dis.numpy()
    stress_yy_dis = stress_yy_dis.numpy()
    stress_xy_dis = stress_xy_dis.numpy()
    scipy.io.savemat('out_f_{}.mat'.format(iStep),{'x': XbndDis[:,0:1], 'y': XbndDis[:,1:2], 
                                         'u': u_delta.numpy(), 'S_xx': stress_xx_dis,
                                         'S_yy': stress_yy_dis})

    
    print("Plot...",file=sys.stderr)
    # plot_field_2d(Xint, state[10], numPtsU, numPtsV,figHeight,figWidth,
    #               title='Damage_'+str(iStep))
    
    # plot_field_2d(Xint, state[9], numPtsU, numPtsV,figHeight,figWidth,
    #               title='Stretch_'+str(iStep))

    # plot_field_2d(XTest, state[16], numPtsU, numPtsV,figHeight,figWidth,
    #               title='F_11_'+str(iStep))
    
    # plot_field_2d(XTest, state[17], numPtsU, numPtsV,figHeight,figWidth,
    #               title='F_12_'+str(iStep))
    
    # plot_field_2d(XTest, state[18], numPtsU, numPtsV,figHeight,figWidth,
    #               title='F_21_'+str(iStep))
    
    # plot_field_2d(XTest, state[19], numPtsU, numPtsV,figHeight,figWidth,
    #               title='F_22_'+str(iStep))
    
    #plot the displacement
    numPtsUTest = len(xPhys)
    numPtsVTest = len(yPhys)

    plot_field_2d(XTest_tf, YTest[:,0], numPtsUTest, numPtsVTest,figHeight,figWidth,
                  title='Computed x-displacement_'+str(iStep))
    plot_field_2d(XTest_tf, YTest[:,1], numPtsUTest, numPtsVTest,figHeight,figWidth,
                  title='Computed y-displacement_'+str(iStep))
    #plot the stresses
    plot_field_2d(XTest_tf, stress_xx_comp, numPtsUTest, numPtsVTest,figHeight,figWidth,
                  title='Computed S_xx_'+str(iStep))
    plot_field_2d(XTest_tf, stress_yy_comp, numPtsUTest, numPtsVTest,figHeight,figWidth,
                  title='Computed S_yy_'+str(iStep))
    # plot_field_2d(XTest_tf, stress_xy_comp, numPtsUTest, numPtsVTest,figHeight,figWidth,
    #               title='Computed sigma_xy_'+str(iStep))
    
    #plot the strains
    # plot_field_2d(XTest_tf, eps_xx_comp, numPtsUTest, numPtsVTest,figHeight,figWidth,
    #               title='Computed strain_xx_'+str(iStep))
    # plot_field_2d(XTest_tf, eps_yy_comp, numPtsUTest, numPtsVTest,figHeight,figWidth,
    #               title='Computed strain_yy_'+str(iStep))
    # plot_field_2d(XTest_tf, eps_xy_comp, numPtsUTest, numPtsVTest,figHeight,figWidth,
    #               title='Computed strain_xy_'+str(iStep))
    
    #Add data points where balance eq is high
    #f_x, f_y = pred_model.balanceEq(xPhysTest, yPhysTest,u_delta,state)
    #f_x = f_x.numpy()
    #f_y = f_y.numpy()
    #indx = heapq.nlargest(3, zip(abs(f_x), itertools.count()))
    #indy = heapq.nlargest(3, zip(abs(f_y), itertools.count()))
    #indx = np.array([indx[0][1],indx[1][1],indx[2][1]])
    #indy = np.array([indy[0][1],indy[1][1],indy[2][1]])
    #if max(abs(f_x)) > 0.5 or max(abs(f_y)) > 0.5:
        #Xint, Yint, state = add_points(indx,indy,Xint,Yint,0.5,Xbnd_all,state,length_phys)
        
    
    model_str = 'model_weights_'+str(iStep)
    pred_model.save_weights(model_str)
    # Update coordinates to new step
    Xint += YTest
    Xbnd[:,:2] += YTest_bnd
    XbndArc[:,:2] += YTest_bnd_arc
    XbndDis[:,:2] += YTest_bnd_dis

plotForceDisp(fdGraph,figHeight,figWidth)
os.chdir(originalDir)
