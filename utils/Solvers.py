#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys
from tensorflow.linalg import inv as inv
from tensorflow.linalg import matmul as matmul
import utils.misc as ms # Needed functions    
import math     
from datetime import datetime
from packaging import version
import tensorboard as tb
 
class Hyperelasticity2D_coll_dist(tf.keras.Model): 
    def __init__(self,layers, train_op, num_epoch, print_epoch, model_data, data_type,length_phys,length_phys_bnd):
        super(Hyperelasticity2D_coll_dist, self).__init__()
        self.model_layers = layers
        self.train_op = train_op
        self.num_epoch = num_epoch
        self.print_epoch = print_epoch
        self.data_type = data_type
        self.Emod = model_data["E"]
        self.nu = model_data["nu"]
        self.adaptive_const_value = tf.constant(1,dtype=tf.float64)
        self.adaptive_const_value_bcs = tf.constant(10,dtype=tf.float64)
        self.adaptive_constant_value_disp = tf.constant(10,dtype=tf.float64)
        self.beta = tf.constant(0.1,dtype=tf.float64)
        self.mu = self.Emod / (2*(1+self.nu))
        self.lam = (self.Emod * self.nu) / ((1+self.nu)*(1-2*self.nu))

        # self.ar = tf.constant(model_data['ar'],dtype=tf.float64)
        # self.l_arc = tf.constant(model_data['l_arc'],dtype=tf.float64)
        # self.l_other = tf.constant(model_data['l_other'],dtype=tf.float64)
        # self.h = tf.constant(model_data['h'],dtype=tf.float64)
        # self.xi_1 = tf.constant(model_data['xi_1'],dtype=tf.float64)
        # self.xi_2 = tf.constant(model_data['xi_2'],dtype=tf.float64)
        if model_data["state"]=="plane strain":
            self.Emat = self.Emod/((1+self.nu)*(1-2*self.nu))*tf.constant([[1-self.nu, self.nu, 0], 
                                                 [self.nu, 1-self.nu, 0], 
                                                 [0, 0, (1-2*self.nu)/2]],dtype=data_type)
        elif model_data["state"]=="plane stress":
            self.Emat = self.Emod/(1-self.nu**2)*tf.constant([[1, self.nu, 0], 
                                                              [self.nu, 1, 0], 
                                                              [0, 0, (1-self.nu)/2]],dtype=data_type)
    #@tf.function                                
    def call(self, X,u_delta):
        uVal, vVal = self.u(X[:,0:1], X[:,1:2],u_delta)
        return tf.concat([uVal, vVal],1)
    
    def dirichletBound(self, X, xPhys, yPhys, u_delta):
        u_val = X[:,0:1]
        v_val = X[:,1:2]
        return u_val, v_val
          
    # Running the model
    #@tf.function
    def u(self, xPhys, yPhys,u_delta):
        X = tf.concat([xPhys, yPhys],1)
        X = 2.0*(X - self.bounds["lb"])/(self.bounds["ub"] - self.bounds["lb"]) - 1.0
        for l in self.model_layers:
            X = l(X)     
            
        # impose the boundary conditions
        u_val, v_val = self.dirichletBound(X, xPhys, yPhys,u_delta)                          
        return u_val, v_val
    
    # Compute the strains
    #@tf.function
    def kinematicEq(self, xPhys, yPhys,u_delta):        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xPhys)
            tape.watch(yPhys)
            u_val, v_val = self.u(xPhys, yPhys,u_delta)
        duxx = tape.gradient(u_val, xPhys)
        duyy = tape.gradient(v_val, yPhys)
        duxy = tape.gradient(u_val, yPhys) 
        duyx = tape.gradient(v_val, xPhys)
        del tape
        return duxx, duyy, duxy, duyx
    
    def green_strain(self, xPhys, yPhys, u_delta):
        duxx, duyy, duxy, duyx = self.kinematicEq(xPhys, yPhys,u_delta) 
        F11 = duxx + 1 
        F12 = duxy
        F21 = duyx
        F22 = duyy + 1
        C11 = F11**2 + F21**2
        C12 = F11*F12 + F21*F22
        C21 = F12*F11 + F22*F21
        C22 = F12**2 + F22**2
        E11 = 0.5 * (C11 - 1)
        E12 = 0.5 * C12
        E21 = 0.5 * C21
        E22 = 0.5 * (C22 - 1)
        return E11, E22, E12
    
    # Compute the stresses
    #@tf.function
    def constitutiveEq(self, xPhys, yPhys,u_delta):
        duxx, duyy, duxy, duyx = self.kinematicEq(xPhys, yPhys,u_delta)     
        F11 = duxx + 1 
        F12 = duxy
        F21 = duyx
        F22 = duyy + 1
        detF = F11 * F22 - F12 * F21
        invF11 = F22 / detF
        invF22 = F11 / detF
        invF12 = -F12 / detF
        invF21 = -F21 / detF
        C11 = F11**2 + F21**2
        C12 = F11*F12 + F21*F22
        C21 = F12*F11 + F22*F21
        C22 = F12**2 + F22**2
        E11 = 0.5 * (C11 - 1)
        E12 = 0.5 * C12
        E21 = 0.5 * C21
        E22 = 0.5 * (C22 - 1)
        P11 = self.mu * F11 + (self.lam * tf.math.log(detF) - self.mu ) * invF11
        P12 = self.mu  * F12 + (self.lam * tf.math.log(detF) - self.mu ) * invF21
        P21 = self.mu  * F21 + (self.lam * tf.math.log(detF) - self.mu ) * invF12
        P22 = self.mu  * F22 + (self.lam * tf.math.log(detF) - self.mu ) * invF22
        S11 = invF11 * P11 + invF12 * P21
        S12 = invF11 * P12 + invF12 * P22
        S21 = invF21 * P11 + invF22 * P21
        S22 = invF21 * P12 + invF22 * P22
        return S11, S22, S12
    
    
    def PIDL_S(self,xPhys,yPhys,u_delta):
        duxx, duyy, duxy, duyx = self.kinematicEq(xPhys, yPhys,u_delta)     
        F11 = duxx + 1 
        F12 = duxy
        F21 = duyx
        F22 = duyy + 1
        detF = F11 * F22 - F12 * F21
        invF11 = F22 / detF
        invF22 = F11 / detF
        invF12 = -F12 / detF
        invF21 = -F21 / detF
        C11 = F11**2 + F21**2
        C12 = F11*F12 + F21*F22
        C21 = F12*F11 + F22*F21
        C22 = F12**2 + F22**2
        
        
        return
    
    # Compute the forces
    #@tf.function
    def balanceEq(self, xPhys, yPhys,u_delta):        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xPhys)
            tape.watch(yPhys)
            stress_xx_val, stress_yy_val, stress_xy_val = self.constitutiveEq(xPhys, yPhys,u_delta)
        dx_stress_xx_val = tape.gradient(stress_xx_val, xPhys)
        dy_stress_yy_val = tape.gradient(stress_yy_val, yPhys)
        dx_stress_xy_val = tape.gradient(stress_xy_val, xPhys)
        dy_stress_xy_val = tape.gradient(stress_xy_val, yPhys)
        del tape
        f_x = dx_stress_xx_val + dy_stress_xy_val
        f_y = dx_stress_xy_val + dy_stress_yy_val
        return f_x, f_y
         
    #Custom loss function
    #@tf.function
    def get_all_losses(self,Xint, Yint, Xbnd, Ybnd,u_delta,XbndArc,YbndArc,XbndDis,YbndDis):
        xPhys = Xint[:,0:1]
        yPhys = Xint[:,1:2]        
        u_bound, v_bound = self.u(XbndDis[:,0:1], XbndDis[:,1:2],u_delta)
        f_x_int, f_y_int = self.balanceEq(xPhys, yPhys,u_delta)  
        int_loss_x = tf.reduce_mean(tf.math.square(f_x_int - Yint[:,0:1]))
        int_loss_y = tf.reduce_mean(tf.math.square(f_y_int - Yint[:,1:2]))
        #int_loss_x = (tf.reduce_sum(self.ar *tf.math.square(f_x_int - Yint[:,0:1]))) * self.xi_1
        #int_loss_y = (tf.reduce_sum(self.ar *tf.math.square(f_y_int - Yint[:,1:2]))) * self.xi_1
        sigma_xx, sigma_yy, sigma_xy = self.constitutiveEq(Xbnd[:,0:1], Xbnd[:,1:2],u_delta)
        trac_x = sigma_xx*Xbnd[:,2:3]+sigma_xy*Xbnd[:,3:4]
        trac_y = sigma_xy*Xbnd[:,2:3]+sigma_yy*Xbnd[:,3:4]
        loss_bnd_tens = tf.where(Xbnd[:,4:5]==0, trac_x - Ybnd, trac_y - Ybnd)
        loss_bnd = tf.reduce_mean(tf.math.square(loss_bnd_tens))
        #loss_bnd = self.l_other * (tf.reduce_sum(tf.math.square(loss_bnd_tens))) * self.xi_2
        sigma_xx_arc, sigma_yy_arc, sigma_xy_arc = self.constitutiveEq(XbndArc[:,0:1], XbndArc[:,1:2],u_delta)
        trac_xarc = sigma_xx_arc*XbndArc[:,2:3]+sigma_xy_arc*XbndArc[:,3:4]
        trac_yarc = sigma_xy_arc*XbndArc[:,2:3]+sigma_yy_arc*XbndArc[:,3:4]
        loss_bnd_arc = tf.where(XbndArc[:,4:5]==0, trac_xarc - YbndArc, trac_yarc - YbndArc)
        loss_bnd_arc_f = tf.reduce_mean(tf.math.square(loss_bnd_arc))
        #loss_bnd_arc = self.l_arc * (tf.reduce_sum(tf.math.square(trac_xarc)) 
        #                           + tf.reduce_sum(tf.math.square(trac_yarc))) * self.xi_2
        #disp_loss = self.l_other * (tf.reduce_sum(tf.math.square(u_bound - YbndDis))) * self.adaptive_constant_value_disp * self.h
        disp_loss = self.adaptive_constant_value_disp*tf.reduce_mean(tf.math.abs(u_bound - YbndDis))
        loss_bnd_total = self.adaptive_const_value_bcs *(loss_bnd + loss_bnd_arc_f)
        return int_loss_x,int_loss_y,loss_bnd_total,disp_loss

    # get gradients
    @tf.function
    def get_grad(self, Xint, Yint, Xbnd, Ybnd,u_delta,XbndArc,YbndArc,XbndDis,YbndDis):
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            L = self.get_loss(Xint, Yint, Xbnd, Ybnd,u_delta,XbndArc,YbndArc,XbndDis,YbndDis)
        g = tape.gradient(L, self.trainable_variables)
        return L, g
    
    #@tf.function
    def tractions(self, Xbnd, u_delta):
        sigma_xx, sigma_yy, sigma_xy = self.constitutiveEq(Xbnd[:,0:1], Xbnd[:,1:2], u_delta)
        trac_x = sigma_xx*Xbnd[:,2:3]+sigma_xy*Xbnd[:,3:4]
        trac_y = sigma_xy*Xbnd[:,2:3]+sigma_yy*Xbnd[:,3:4]
        return trac_x, trac_y
    
    #@tf.function
    def get_loss(self,Xint, Yint, Xbnd, Ybnd,u_delta,XbndArc,YbndArc,XbndDis,YbndDis):
        losses = self.get_all_losses(Xint, Yint, Xbnd, Ybnd,u_delta,XbndArc,YbndArc,XbndDis,YbndDis)
        return sum(losses)


    #@tf.function
    def adaptive_const(self,Xint, Yint, Xbnd, Ybnd,u_delta,XbndArc,YbndArc,XbndDis,YbndDis):
        grad_resx = []
        grad_resy = []
        grad_bcs_disp = []
        grad_bcs = []
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.trainable_variables)      
            l1,l2,l3,l4 = self.get_all_losses(Xint, Yint, Xbnd, Ybnd,u_delta,XbndArc,YbndArc,XbndDis,YbndDis)
        for i in range(len(self.layers) - 1 ):
            grad_resx.append(tape.gradient(l1,self.trainable_variables[i])[0])
            grad_resy.append(tape.gradient(l2,self.trainable_variables[i])[0])
            grad_bcs.append(tape.gradient(l3,self.trainable_variables[i])[0])
            grad_bcs_disp.append(tape.gradient(l4,self.trainable_variables[i])[0])
        del tape
        max_grad_resx_list = []
        max_grad_resy_list = []
        mean_grad_bcs_list = []
        mean_grad_bcs_disp_list = []
        
        for i in range(len(self.layers)-1):
            max_grad_resx_list.append(tf.reduce_max(tf.abs(grad_resx[i])))
            max_grad_resy_list.append(tf.reduce_max(tf.abs(grad_resy[i])))
            mean_grad_bcs_list.append(tf.reduce_mean(tf.abs(grad_bcs[i])))
            mean_grad_bcs_disp_list.append(tf.reduce_mean(tf.abs(grad_bcs_disp[i])))
            
        max_grad_resx = tf.reduce_max(tf.stack(max_grad_resx_list))
        max_grad_resy = tf.reduce_max(tf.stack(max_grad_resy_list))
        mean_grad_bcs = tf.reduce_mean(tf.stack(mean_grad_bcs_list))
        mean_grad_bcs_disp = tf.reduce_mean(tf.stack(mean_grad_bcs_disp_list))
        max_grad_res = max_grad_resx + max_grad_resy
        
        adaptive_constant_bcs = max_grad_res / mean_grad_bcs
        adaptive_constant_disp = max_grad_res / mean_grad_bcs_disp
        return adaptive_constant_bcs, adaptive_constant_disp          
    
    def update_adaptive_const(self, Xint, Yint, Xbnd, Ybnd,u_delta,XbndArc,YbndArc,XbndDis,YbndDis):
        adaptive_const_bcs, adaptive_constant_disp = self.adaptive_const(Xint, Yint, Xbnd, Ybnd,
                                                                         u_delta,XbndArc,YbndArc,XbndDis,YbndDis)
        self.adaptive_const_value_bcs = adaptive_const_bcs * (1.0 - self.beta) \
                                      + self.beta * self.adaptive_const_value_bcs
        self.adaptive_constant_value_disp = adaptive_constant_disp * (1.0 - self.beta)\
                                       + self.beta * self.adaptive_constant_value_disp 
        return

    def obtain_adaptive_const(self, Xint, Yint, Xbnd, Ybnd,u_delta,XbndArc,YbndArc,XbndDis,YbndDis):
        return self.adaptive_const_value_bcs, self.adaptive_constant_value_disp
  
    # perform gradient descent
    def network_learn(self,Xint,Yint, Xbnd, Ybnd, u_delta,XbndArc,YbndArc,XbndDis,
                      YbndDis):
        xmin = tf.math.reduce_min(Xbnd[:,0])
        ymin = tf.math.reduce_min(Xbnd[:,1])
        xmax = tf.math.reduce_max(Xbnd[:,0])
        ymax = tf.math.reduce_max(Xbnd[:,1])
        self.bounds = {"lb" : tf.reshape(tf.stack([xmin, ymin], 0), (1,2)),
                       "ub" : tf.reshape(tf.stack([xmax, ymax], 0), (1,2))}
        for i in range(self.num_epoch):
            L, g = self.get_grad(Xint, Yint, Xbnd, Ybnd, u_delta,XbndArc,YbndArc,XbndDis,YbndDis)
            self.train_op.apply_gradients(zip(g, self.trainable_variables))
            if i%self.print_epoch==0:
                int_loss_x, int_loss_y, loss_bond, disp_loss = self.get_all_losses(Xint, Yint, Xbnd, Ybnd,u_delta,
                                                                        XbndArc,YbndArc,XbndDis,YbndDis)
                #self.update_adaptive_const(Xint, Yint, Xbnd, Ybnd,u_delta,XbndArc,YbndArc,XbndDis,YbndDis)
                L = int_loss_x + int_loss_y + loss_bond + disp_loss
                print("Epoch {} loss: {}, int_loss_x: {}, int_loss_y: {}, bnd_loss: {}, disp_loss: {}".format(i, 
                                                                    L, int_loss_x, int_loss_y, loss_bond,disp_loss),file=sys.stderr)
                
                print("Epoch {}, Adaptive_boundary: {}, Adaptive_displacement: {}".format(i,
                                        self.adaptive_const_value_bcs,self.adaptive_constant_value_disp),file=sys.stderr)
            if L < 0.0001:
                break
