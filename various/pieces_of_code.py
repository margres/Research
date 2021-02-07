#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 10:49:34 2020

@author: mrgr
"""

 
    text=np.arange(tau_list[-1],10,np.diff(t_new)[0])
    asymptote=geom_optics(text) 
    mask_asymptote=Ft_new<Ft_new
    plt.plot(text,asymptote, label='geom')
    #plt.legend() 
    #plt.show()
    
    Ft_new = np.concatenate([Ft_list,asymptote])
    t_new= np.concatenate([tau_list,text])
    #plt.plot(t_new,Ft_new, label='final')
    
    
    
    
    
    
    #i want to extend using the geometrical approximation, in order to do that i need to find
    #where the 2 curves encounter. to find that i also need to extend with the logaritmic fit
 
    text=np.arange(tau_list[-1],150,np.diff(t_new)[-1])
    asymptote=np.zeros_like(text)
    for T,m in zip(tauI, muI):
        asymptote+=geom_optics(T,m,text) 
    asymptote+=0.5*np.sqrt(xL1**2.+xL2**2)
    #Ftmp_asy=geom_optics(t_new) 
    #mask_asymptote=Ft_new<Ftmp_asy &
    
    
    index_ext=np.where(t_new==tau_list[-1])[0][0]
    idx = np.argwhere(np.diff(np.sign(asymptote - Ft_new[index_ext:]))).flatten()[0]
    print(idx)
    
    plt.axvline(text[idx])
    mask_asymptote=text>text[idx]
    
    Ft_new = np.concatenate([Ft_new[:(index_ext+idx)],asymptote[mask_asymptote]])
    t_new= np.concatenate([t_new[:(index_ext+idx)],text[mask_asymptote]])
    print(len(t_new),len(Ft_new))