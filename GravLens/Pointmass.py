#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 10:09:37 2020

@author: mrgr
"""


import numpy as np 
import scipy.special as sc
from mpmath import hyp1f1
import matplotlib.pyplot as  plt
import cmath
import pandas as pd
import glob

def Pointmass(w,y):
     
    
    xm=(y+(y**2 + 4)**(1/2))/2
    phim=(xm-y)**2/2 - np.log(xm)
    expon= np.exp(np.pi*w/4 + 1j*w/2* (np.log(w/2) -2* phim ) )
    #print(w)
    Fan=expon*sc.gamma(1-1j/2*w)*hyp1f1(1j/2*w, 1, 1j/2*w*y**2 )
    F.append(Fan)
            
    return F

if __name__ == '__main__':
    
    #y : distance from the lens 
    y_range=[np.sqrt(2*0.1**2)]

    for y in y_range:
        w_range=np.linspace(0.001,100,2000)
        #w_range=np.logspace(np.log(0.001),np.log(100),550)
        #print(str(y))
        for w in w_range:
            print(str(w))
            F = Pointmass(w,y)
            
            Famp=[float(abs(complex(i))) for i in F]
            Fphase=[float(cmath.phase(complex(i))) for i in F]
            
            df = pd.DataFrame(list(zip(Famp,Fphase,w_range)),columns=['Famp','Fphase','w'] )
            
            add_info='pointmass_lens_dist_'+str(y)
            df.to_csv('./Analytic_'+add_info+'.txt', sep='\t')
    
    plot=False
    if plot==True:
        ##### Plot ####
        folders='./Levin/Analytic_*'
        folders_list=sorted(glob.glob(folders))  
        
        w_range=np.linspace(0.001,100,5000)
        
        for fol,y in zip(folders_list, y_range):
            
            dfpoint=pd.read_csv(fol, sep="\t")
            amp=dfpoint.Famp.values
            phase=dfpoint.Fphase.values
        
            plt.plot(w_range,amp,'-',label='y='+str(y))
            plt.xscale('log')
            plt.yscale('log')
            plt.legend()
            plt.savefig('amp_analytic_pointmass.png')
            plt.show()
            
            
            plt.plot(w_range,phase)
            plt.xscale('log')
            plt.legend()
            plt.savefig('phase_analytic_pointmass.png')
            plt.show()
    