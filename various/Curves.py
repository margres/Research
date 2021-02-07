#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 12:33:09 2020

@author: mrgr

Functions to plot critical curves and caustics
"""
import scipy.optimize as op
import numpy as np 

def DistMatrix( dpsid11,dpsid22,dpsid12, kappa,gamma):
    
    # Jacobian matrix
    j11 = 1. - kappa - gamma - dpsid11
    j22 = 1. - kappa + gamma - dpsid22
    j12 = -dpsid12
    detA=j11*j22-j12*j12
    
    return detA
    


def PseudoSIS(rt,theta_t,kappa,gamma,fact, caustics=False):
    
    a=fact[0]
    b=fact[1]
    c=fact[2]
    
    x1 = rt*np.cos(theta_t)
    x2 = rt*np.sin(theta_t)

    #x1=x[0]
    #x2=x[1]
    
    #first derivative

    dx12dx22 = np.sqrt((x1**2.+x2**2.)/c**2 + b**2)
    dpsi1 = a*x1/dx12dx22/c**2
    dpsi2 = a*x2/dx12dx22/c**2
    
    if caustics==True:
        
        x1_per = x1*(kappa+gamma)
        x2_per = x2*(kappa-gamma)
        
        return np.column_stack((x1_per+dpsi1, x2_per+dpsi2))
        #return npdpsi1,dpsi2
    
    
    
    #second derivatives 
    
    dx12pdx22_32 = ((x1**2.+ x2**2.)/c**2+b**2)**(3/2)
    # d^2psi/dx1^2
    dpsid11 = a*(x2**2+b**2*c**2)/dx12pdx22_32/c**4
    # d^2psi/dx2^2
    dpsid22 = a*(x1**2+b**2*c**2)/dx12pdx22_32/c**4
    # d^2psi/dx1dx2
    dpsid12 = -a*(x1*x2)/dx12pdx22_32/c**4

    detA = DistMatrix(dpsid11,dpsid22,dpsid12, kappa,gamma)
    
    return detA

def PowerLaw(rt,theta_t, kappa, gamma,fact,caustics=False):
    
    x1 = rt*np.cos(theta_t)
    x2 = rt*np.sin(theta_t)
    
    E_r = 1.
    #x1=x[0]
    #x2=x[1]
    p=fact[3]
    
    dpsi1=x1*E_r**(2 - p)*(x1**2+x2**2)**(p/2 - 2)
    dpsi2=x2*E_r**(2 - p)*(x1**2+x2**2)**(p/2 - 2)
    
    if caustics==True:
        
        x1_per = x1*(kappa+gamma)
        x2_per = x2*(kappa-gamma)
        
        return np.column_stack((x1_per+dpsi1, x2_per+dpsi2))
        
    dpsid11=E_r**(2 - p)*(x1**2+x2**2)**(p/2 - 2)*((p - 1)*x1**2+x2**2)
    dpsid22=E_r**(2 - p)*(x1**2+x2**2)**(p/2 - 2)*((p - 1)*x2**2+x1**2)
    dpsid12=(p - 2)*x1*x2*E_r**(2 - p)*(x1**2+x2**2)**(p/2 - 2)
    
    detA = DistMatrix(dpsid11,dpsid22,dpsid12, kappa,gamma)
    
    return detA


def PointMass(rt,theta_t,kappa, gamma, fact,caustics=False):
    
    x1 = rt*np.cos(theta_t)
    x2 = rt*np.sin(theta_t)
    
    #x1=x[0]
    #x2=x[1]
    
    #first derivative
    
    dx12dx22 = x1**2.+x2**2
    dpsi1 = x1/dx12dx22
    dpsi2 = x2/dx12dx22
    
    if caustics==True:
        x1_per = x1*(kappa+gamma)
        x2_per = x2*(kappa-gamma)
        
        return  np.column_stack((x1_per+dpsi1, x2_per+dpsi2))
        #return np.column_stack((dpsi1,dpsi2))

    
    #second derivatives 
    
    dx22mdx12 = x2**2.-x1**2.
    dx12pdx22_2 = (x1**2.+x2**2.)**2.
    
    # d^2psi/dx1^2
    dpsid11 = dx22mdx12/dx12pdx22_2
    # d^2psi/dx2^2
    dpsid22 = -dpsid11
    # d^2psi/dx1dx2
    dpsid12 = -2*x1*x2/dx12pdx22_2
    
    detA = DistMatrix(dpsid11,dpsid22,dpsid12, kappa,gamma)
    
    return detA

def LensEq(rt,theta_t,kappa, gamma, fact, lens_model):
    
    x1 = rt*np.cos(theta_t)
    x2 = rt*np.sin(theta_t)
    
    alpha=eval(lens_model)(rt,theta_t,kappa, gamma, fact, caustics=True)
    beta=np.column_stack((x1,x2))-alpha
    '''
    print(x[0][:10])
    print(alpha[:10])
    print(beta[:10])
    '''
    return beta

def FindCrit(lens_model,kappa, gamma, fact):
    
    N_theta = 800
    theta_t = np.linspace(0,2*np.pi,N_theta)  
    
    rt_res = []
    
    N_r = 100
    rmin = 0.001
    rmax = 5.
    rt = np.linspace(rmin,rmax,N_r)
    
    for i in range(N_theta):
    	for j in range(N_r-1):
    		if eval(lens_model)(rt[j],theta_t[i],kappa, gamma,fact)* eval(lens_model)(rt[j+1],theta_t[i],kappa, gamma,fact)<=0:
    			tmp = op.brentq(eval(lens_model), rt[j],rt[j+1], args=(theta_t[i],kappa, gamma,fact))
    			rt_res.append(tmp)
    
    return rt_res, theta_t


def PlotCurves(xS12,xL12,kappa,gamma,lens_model,fact):
    
    fig = plt.figure(dpi=100)
    xL1 = xL12[0]
    xL2 = xL12[1]
    
    xS1 = xS12[0]
    xS2 = xS12[1]
    
    N_ = 800
    theta_t = np.linspace(0,2*np.pi,N_)  
        
    #N_r = 100
    rmin = 0.001
    rmax = 5.
    rt = np.linspace(rmin,rmax,N_)
    
    for r in r_t:
        x1.append(r*np.cos(theta_t))
        x2.append(r*np.sin(theta_t))
    
    #xy_lin=np.linspace(-5,5,1000)
    #X,Y = np.meshgrid(xy_lin, xy_lin)
    
    
    crit_curv=eval(lens_model)(rt, theta_t, kappa, gamma, fact)
    cp = plt.contour((x1, x2), crit_curv,[0], colors='k',linestyles= '-', linewidths=1) 
    plt.show()
    
    '''
    #PseudoSIS(rt,theta_t,kappa,gamma,fact, caustics=False):
    
    #I get the coordinates of the contour plot
    xyCrit_all = cp.collections[0].get_paths() 
    
    for i in range(np.shape(xyCrit_all)[0]):
        
        #xyCrit = cp.collections[0].get_paths()[i] 
        xyCrit = xyCrit_all[i].vertices  
        xCrit=xyCrit[:,0] 
        yCrit=xyCrit[:,1]
        plt.plot(xCrit,yCrit,'--', color='k',label='critical curves',linewidth=0.7)
        #exec( "xyCrit_"+str(i)+"= xyCrit.vertices")
        #xyCrit = xyCrit.vertices    
        xyCaus=LensEq((xCrit,yCrit), kappa, gamma, fact, lens_model)
        plt.plot(xyCaus[:,0],xyCaus[:,1], 'k--',label='caustics',linewidth=0.7) 
    
    plt.scatter(xL1, xL2, marker='x',color='r', label='lens')
    plt.scatter(xS1, xS2, marker='*',color='orange', label='source')

    plt.show()
    '''
if __name__ == '__main__':
    
    import matplotlib.pyplot as plt

    
    kappa=0
    gamma=0
    lens_model='PseudoSIS'
    fact=[1,0.5,1]
    
    xL1=0
    xL2=0
    xL12=[xL1,xL2]
    
    xS1=0
    xS2=0.5
    xS12=[xS1,xS2]
    
    a=1
    b=0.5
    c=1
    p=1
    #PlotCurves(xS12,xL12,kappa,gamma,lens_model, [a,b,c,p])
    fact=[a,b,c,p]
    if True:
    
        '''
        
        Alternative method to find critical curves and caustics
        
        '''
        
        rt_res, theta_t=FindCrit(lens_model,kappa, gamma, fact)    
        plt.gca().set_aspect('equal', adjustable='box')
        
        
        
        cosTheta_t = np.cos(theta_t)
        sinTheta_t = np.sin(theta_t)
        
        xcr = np.zeros_like(theta_t)
        ycr = np.zeros_like(theta_t)
        
        xcau = np.zeros_like(theta_t)
        ycau = np.zeros_like(theta_t)
    
        for i in range(theta_t.size):
            	xcr[i] = rt_res[i]*cosTheta_t[i]#+xl
            	ycr[i] = rt_res[i]*sinTheta_t[i]# +yl
            
            	tmp = LensEq(rt_res[i],theta_t[i],kappa, gamma, fact, lens_model)
            	xcau[i] = tmp[:,0]
            	ycau[i] = tmp[:,1]
                
        plt.plot(xcr,ycr,'.',color = 'blue', label='critical')
        plt.plot(xcau,ycau,'.',color = 'red', label='caustic')
        
        plt.scatter(xL1, xL2, marker='x',color='r', label='lens')
        plt.scatter(xS1, xS2, marker='*',color='orange', label='source')
        plt.legend()
    
    
   
