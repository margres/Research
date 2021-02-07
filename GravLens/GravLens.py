#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 17:37:54 2021

@author: mrgr
"""

if __name__ == '__main__':
    
    import numpy as np
    from Levin.Levin import LevinMethod
    from HistCounting import HistMethod
    
    yL1=0.3
    yL2=0 
    yL=[yL1,yL2]
        
    y=0.1
    kappa=0
    gamma=0
    
    w_range=np.round(np.linspace(0.001,100,1050),5)
        
    a=1 #amplitude parameter
    b=0.5 #[0,0.25,0.5,0.75,1,1.5]#core
    c=1 #flattening parameter
    p=1 #poewer law value
    fact=[a,b,c,p]
    onedim=True
    
    if onedim=='True':
        lens_model='SIS'
        models=['point','SIS','SIScore','softenedpowerlaw','softenedpowerlawkappa']   
        LevinMethod(w_range,y, lens_model, fact)
    
    else:
        lens_model='point'
        models=['point','SIS']    
        HistMethod(yL,kappa,gamma, lens_model)
        