#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 12:33:09 2020

@author: mrgr

Functions to plot critical curves and caustics
"""
#import scipy.optimize as op
import numpy as np 
import matplotlib.pyplot as plt


def DistMatrix( dpsid11,dpsid22,dpsid12, kappa,gamma):
    
    # Jacobian matrix
    j11 = 1. - kappa - gamma - dpsid11
    j22 = 1. - kappa + gamma - dpsid22
    j12 = -dpsid12
    detA=j11*j22-j12*j12
    
    return detA
    


def SIScore(x12,kappa,gamma,fact, caustics=False):
    
    a=fact[0]
    b=fact[1]
    

    x1=x12[0]
    x2=x12[1]
    
    #psi= a*sqrt(x1**2+x2**2+b**2)
    
    #first derivative
    dx12dx22 = np.sqrt(x1**2.+x2**2.+b**2.)
    dpsi1 = a*x1/(dx12dx22)
    dpsi2 = a*x2/(dx12dx22)
    
    if caustics==True:
        
        x1_per = x1*(kappa+gamma)
        x2_per = x2*(kappa-gamma)
        
        return np.column_stack((x1_per+dpsi1, x2_per+dpsi2))
        #return npdpsi1,dpsi2
    
    
    
    #second derivatives 
    
    dx12pdx22_32 = (x1**2.+ x2**2.+b**2)**(3/2)
    # d^2psi/dx1^2
    dpsid11 = (b**2+x2**2)/dx12pdx22_32
    # d^2psi/dx2^2
    dpsid22 = (b**2+x1**2)/dx12pdx22_32
    # d^2psi/dx1dx2
    dpsid12 = -x1*x2/dx12pdx22_32
    
    detA = DistMatrix(dpsid11,dpsid22,dpsid12, kappa,gamma)
    
    return detA



def point(x12,kappa, gamma, fact,caustics=False):
    
    x1 = x12[0] 
    x2 = x12[1] 
    
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

def softenedpowerlaw(x12,kappa, gamma, fact,caustics=False):
    
    
    a=fact[0]
    b=fact[1]
    c=fact[2]
    p=fact[3]
    
    x1 = x12[0] 
    x2 = x12[1] 
    

    dpsi1=a*p*x1*(b**2+(x1**2+x2**2)/c**2)**(p/2 - 1)/c**2
    dpsi2=a*p*x2*(b**2+(x1**2+x2**2)/c**2)**(p/2 - 1)/c**2
    
    if caustics==True:
        #perturbation contribution
        x1_per = x1*(kappa+gamma)
        x2_per = x2*(kappa-gamma)
        
        return  np.column_stack((x1_per+dpsi1, x2_per+dpsi2))
        
    #second derivatives 
    
    ppart=(b**2+ (x1**2+x2**2))    
    dpsid11=a*p*ppart**(p/2. - 1.) + 2*a*(p/2. - 1.)*p*x1**2.*ppart**(p/2. - 2.)
    dpsid22=a*p*ppart**(p/2. - 1.) + 2*a*(p/2. - 1.)*p*x2**2.*ppart**(p/2. - 2.)
    dpsid12=2*a*ppart**(p/2. - 2.)*p*x1*x2*(p/2. - 1.)
    
    detA = DistMatrix(dpsid11,dpsid22,dpsid12, kappa,gamma)
    
    return detA
    

def softenedpowerlawkappa(x12,kappa, gamma, fact,caustics=False):
    
    a,b,p =fact[0],fact[1],fact[3]
    
    x1 = x12[0] 
    x2 = x12[1] 
    

    x1x22= x1**2+x2**2
    
    #if  
    
    if p==0:
        dpsi1= (a**2*x1*np.log((b**2 + x1x22)/b**2))/x1x22
        dpsi2= (a**2*x2*np.log((b**2 +  x1x22)/b**2))/x1x22
        dpsid11= (2*a**2*x1**2)/(x1x22*(b**2 + x1x22)) - (a**2*(x1**2 - x2**2)*np.log(1 + x1x22/b**2))/x1x22**2
        dpsid22= (2*a**2*x2**2)/(x1x22*(b**2 + x1x22)) - (a**2*(-x1**2 + x2**2)*np.log(1 + x1x22/b**2))/x1x22**2
        dpsid12= (2*a**2*x1*x2*(x1x22/(b**2 +x1x22) - np.log(1 + x1x22/b**2)))/x1x22**2
    else:
        
        dpsi1= (x1*a**(2 - p)*((b**2 + x1x22)**(p/2) - b**p))/(p *x1x22)
        dpsi2= (x2*a**(2 - p)*((b**2 + x1x22)**(p/2) - b**p))/(p *x1x22)
        #dpsid11= (a**(2 - p)*b**p*(x1**2 - x2**2))/(p*x1x22**2) + (a**(2 - p)*x1x22**(p/2 - 2)* (-b**2*x1**2 + b**2*x2**2 + p*x1**4 + p*x1**2*x2**2 - x1**4 + x2**4)*(b**2/x1x22 + 1)**(p/2))/(p*(b**2 + x1x22))
        dpsid11=(a**(2 - p) *((x1x22)**(p/2)*(b**2*(x2**2 - x1**2) + (p - 1)*x1**4 + p*x1**2*x2**2 + x2**4)*(b**2/(x1x22)+1)**(p/2) + b**p*(x1 - x2)*(x1 + x2)*(b**2 + x1**2 + x2**2)))/(p*(x1x22)**2*(b**2 + x1x22)) 
        dpsid22=(a**(2 - p)*((x1x22)**(p/2)*(b**2*(x1 - x2)*(x1 + x2) + p*x1**2*x2**2 + (p-1)*x2**4+x1**4)*(b**2/(x1**2+x2**2)+1)**(p/2)-b**p*(x1-x2)*(x1+x2)*(b**2+x1x22)))/(p*x1x22**2*(b**2+x1**2+x2**2))
        #dpsid22= (a**(2 - p)*b**p*(x2**2 - x1**2))/(p*x1x22**2) + (a**(2 - p)*x1x22**(p/2 - 2)* (b**2*x1**2 - b**2*x2**2 + p*x1**2*x2**2 + p*x2**4 + x1**4 - x2**4) *(b**2/x1x22 + 1)**(p/2))/(p*(b**2 + x1x22))
        #dpsid12= (x1*x2*a**(2 - p)*(((p - 2)*(x1x22) - 2*b**2)*(b**2 + x1x22)**(p/2) + 2*b**p*(b**2 + x1**2 + x2**2)))/(p*x1x22**2*(b**2 + x1x22))
        dpsid12=(x1*x2*a**(2-p)*(((p - 2)*x1x22 - 2*b**2)*(b**2 + x1x22)**(p/2) + 2 *b**p *(b**2 + x1x22)))/(p*x1x22**2*(b**2 + x1x22))
        
        
     
    if caustics==True:
        #perturbation contribution
        x1_per = x1*(kappa+gamma)
        x2_per = x2*(kappa-gamma)
        
        return  np.column_stack((x1_per+dpsi1, x2_per+dpsi2))
    
  
    
    detA = DistMatrix(dpsid11, dpsid22, dpsid12, kappa,gamma)
    
    return detA
    

def LensEq(x12,kappa, gamma, fact, lens_model):
    
    x1 = x12[0] 
    x2 = x12[1]
    E_r=1
    
    alpha=eval(lens_model)((x1,x2),kappa, gamma, fact, caustics=True)
    beta=np.column_stack((x1,x2))-E_r*alpha
    
    '''
    print(x[0][:10])
    print(alpha[:10])
    print(beta[:10])
    '''
    
    return beta


def PlotCurves(xS12,xL12,kappa,gamma,lens_model,fact):
    
    xL1 = xL12[0]
    xL2 = xL12[1]
    
    xS1 = xS12[0]
    xS2 = xS12[1]   
    
    a=fact[0]
    b=fact[1]
    c=fact[2]
    p=fact[3]
    
    shear='$\gamma$='+str(gamma)+' $\kappa$='+str(kappa)
    
    if lens_model=='softenedpowerlaw' and p>1.7:
        xy_lin=np.linspace(-500,500,1000)
    else:    
        xy_lin=np.linspace(-10,10,1000)
    X,Y = np.meshgrid(xy_lin, xy_lin)
    
    crit_curv=eval(lens_model)((X,Y), kappa, gamma, fact)
    
    plt.figure(dpi=100)

    plt.tight_layout()
    plt.rcParams["figure.figsize"] = (6,6)
    params = {'axes.labelsize': 16,
              'axes.titlesize': 16,
              'xtick.labelsize' : 16,
              'ytick.labelsize' : 16,
              'font.size':16,
              'legend.fontsize':12,
              'lines.markersize':6
             }
    plt.rcParams.update(params)
    
    cp = plt.contour(X,Y, crit_curv,[0], colors='k',linestyles= '-', linewidths=0.1)     
    
    #I get the coordinates of the contour plot
    xyCrit_all = cp.collections[0].get_paths() 
    
    print(np.shape(xyCrit_all))
    for i in range(np.shape(xyCrit_all)[0]):
        figLim=0
        #xyCrit = cp.collections[0].get_paths()[i] 
        xyCrit = xyCrit_all[i].vertices  
        xCrit=xyCrit[:,0] 
        yCrit=xyCrit[:,1]
        plt.plot(xCrit,yCrit,'k--',linewidth=0.7)
    
        xyCaus=LensEq((xCrit,yCrit), kappa, gamma, fact, lens_model)
       
        #print(xyCaus)
        if xyCaus[:,0].any() <1e5 and xyCaus[:,1].any() <1e5 :
            plt.scatter(0,0, s=15, c='k', marker='o')
        
        plt.plot(xyCaus[:,0],xyCaus[:,1],'k-',linewidth=0.7) 
        #plt.title('p='+str(p))
        if lens_model!='point' and lens_model!='SIScore':
            plt.title(str(lens_model)+' - b='+str(b)+' p='+str(p))
        elif lens_model=='SIScore' : 
            plt.title(str(lens_model)+' - b='+str(b))
        #plt.title('SIS with external shear')
       
        tmp=np.max(xyCrit_all[i].vertices)
        if tmp>figLim:
            figLim=tmp
            
    plt.plot(np.nan,np.nan,'k--',label='critical curves',linewidth=0.7)
    plt.plot(np.nan,np.nan,'k-',label='caustics',linewidth=0.7) 
    
    if gamma !=0 or kappa!=0:
        plt.figtext(.5, .2, shear, fontsize=12)
    #plt.title('point')
    cp.ax.set_ylabel('$x_2$', fontsize=16)
    cp.ax.set_xlabel('$x_1$', fontsize=16)
    
    #I always plot the critical curves and caustics with the lens in the middle
    
    if xL1!=0 and xL2!=0:
        xS1=xS1-xL1
        xS2=xS2-xL1
        xL1=0
        xL2=0
    
    plt.scatter(xL1, xL2, marker='x',color='r', label='lens')
    
    try:
        for x1,x2 in zip(xS1,xS2):
            plt.scatter(x1, x2, marker='*')
        plt.plot(np.nan,np.nan, marker='*',label='source', color='k', linestyle='None')
    except:
        plt.scatter(xS1, xS2, marker='*',color='tab:blue', label='source')
    
    plt.xlim(-figLim-1, figLim+1)
    plt.ylim(-figLim-1, figLim+1)
    plt.legend()
    
    try:
        y =round((xS1**2+xS2**2)**(0.5),2)
    except:
        y='various'

    if lens_model=='softenedpowerlaw':        
        add_info=lens_model+'_lens_dist_'+str(y)+'_a_'+str(a)+'_b_'+str(b)+ '_p_'+str(p)          
    
    elif lens_model=='softenedpowerlawkappa':        
        add_info=lens_model+'_lens_dist_'+str(y)+'_a_'+str(a)+'_b_'+str(b) + '_p_'+str(p)   
    
    else:
        add_info=lens_model+'_lens_dist_'+str(y)+'_a_'+str(a)+'_b_'+str(b)+'_k_'+str(kappa)+ '_g_'+ str(gamma)
            
    plt.tight_layout()    
    plt.savefig('../plot/'+lens_model+'/CC_'+add_info+'.png')
    plt.show()
    
    
    
if __name__ == '__main__':
    
    kappa=0
    gamma=0
    lens_model='softenedpowerlawkappa'
    
    models=['point','SIScore','softenedpowerlaw','softenedpowerlawkappa']
    
    xL1=0
    xL2=0
    xL12=[xL1,xL2]
    
    
    xS2=0.3
    #xS2=np.append(0.1,xS2)
    #xS2=np.array([ 0.25, 0.5 , 0.75, 1 ])


    #xS2=np.sqrt(0.1**2 + 0.1**2)  #np.array([0.1,0.25,0.5,1,3]) 

    #[0.25,0.5,0.75,1,3]
   
    xS1=np.zeros_like(xS2)
    xS12=[xS1,xS2]
    
    a = 1
    b = 0.5
    c = 1
    p = -1

    fact = [a,b,c,p]
    
    #bLin=np.linspace(0,1.5,7)
    #for b in bLin:
    #    fact= [a,b,c,p]
    PlotCurves(xS12,xL12,kappa,gamma,lens_model, fact)
    #plt.show()

    
    
    
   
