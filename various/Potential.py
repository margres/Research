#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 16:23:07 2020

@author: mrgr
"""

import numpy as np  
import matplotlib.pyplot as plt

def PlotPotential(fact, model_lens):
                  
    xy_lin=np.linspace(-10,10,100)
    X,Y = np.meshgrid(xy_lin, xy_lin)
    r=np.sqrt(X**2+Y**2)
    
    if model_lens == 'SIS':
        
        kappa=(r**2)**(-1/2)
        
    elif model_lens== 'SIScore':
        
        a,b,c=fact[0],fact[1],fact[2]
        kappa=a*(b**2+r**2/c**2)**(-1/2)
        
    elif model_lens == 'softenedpowerlawkappa':
        
        #isothermal power law for p=0
        #modified Hubble model for p= 0
        #Plummer model for p =-2
        
        a,b,c,p=fact[0],fact[1],fact[2], fact[3]       
        kappa=0.5*a**(2-p)/(b**2+r**2)**(1-p/2)
        
    '''
    elif model_lens== 'point':
        pot=np.log(abs(x))
        
    
    elif model_lens == 'powerlaw':
        
        E_r=1
        p=fact[3] #p=1 for SIS
        const=(E_r**(2-p)/p)
        pot=const*x**p
      
     
    elif model_lens == 'softenedpowerlaw':
        
        a,b,c=fact[0],fact[1],fact[2]
        p=fact[3]
        pot=a*(x**2/c**2+b**2)**(p/2) - a*b**p
    ''' 
        
    plt.imshow(kappa,  cmap=plt.get_cmap('hot'), interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar()
    plt.show()
    
    
    
if __name__ == '__main__':
    
    PlotPotential([1,0,1,2], 'softenedpowerlawkappa')
    
    '''
    xy_lin=np.linspace(-2,2,1000)
    X,Y = np.meshgrid(xy_lin, xy_lin)
    r=np.sqrt(X**2+Y**2)
    s=0.5
    rho=(s**2+r**2)**(-1/2)
    plt.imshow(rho)
    '''
    #plt.scatter()