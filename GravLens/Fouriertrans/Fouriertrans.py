
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 16:59:04 2020

@author: mrgr

###### script to calculate the magnification factor

#### it uses the FFT and the analytical FT




xs -> x-axis coordinates from the fit, time
ys -> y-axis coordinates from the fit, F
"""

import numpy as np
import numpy.polynomial.polynomial as poly
import scipy.interpolate as si
import matplotlib.pyplot as plt




def FFT(xs,ys, add_zeros='False'):
    
    '''
    It returns the FT of the ys values. 
    '''
    
    dt= xs[1]-xs[0]
    '''
    if add_zeros=='True':
        
        N_zeros = 1000
        t_tail = np.linspace(xs[-1]+dt, xs[-1]+dt*N_zeros, N_zeros)
        t_head = np.linspace(xs[0]-N_zeros*dt, xs[0]-dt, N_zeros)
        t_final = np.concatenate([t_head, xs, t_tail])
        Ftd_final = np.concatenate([np.zeros(N_zeros), ys, np.zeros(N_zeros)])
        N = len(t_final)
    else:
   
        N=len(xs)
        t_final=xs
        Ftd_final=ys
    '''
    N=ys.size +1000
    # 4. FFT
    ## note: Ftd_final is real, so first half of the FFT series gives usable information
    ##      you can either remove the second half of ifft results, or use a dedicated function ihfft   
    Fw=np.fft.ihfft(ys, n=N) # I can add parameter n with value higher tan len(Ftd_final) and it will automatically create a padding of zeros
    ## multiply back what is divided
    Fw *= N
    ## multiply sampling interval to transfer sum to integral
    Fw *= dt
    freq = np.fft.rfftfreq(N,d=dt)
    w = freq*2.*np.pi
    
    return w[1:],Fw[1:]


def FT_clas(freq,T,mu,typeI):
    
    '''
    semi-calssical analytical contribution, eq. 34 and 39 -- Ulmer's paper 

    '''   
    if typeI=='saddle':
        n=0.5
        #saddle point
        #return abs(mu)**0.5* 1j*np.exp(1j*freq*T)*abs(mu)**0.5

    elif typeI=='min':   
        #min
        n=0
        #return abs(mu)**0.5*np.exp(1j*freq*T- 1j*np.pi*T)
    
    elif typeI=='max':
        n=1
        
    return abs(mu)**0.5*np.exp(1j*freq*T- 1j*np.pi*n)
        
    

def Fd_w(t,F,t_ori,Ft_ori):
    
    '''
    eq. 5.6  
    
    Note that t_final and Ftd_final may contain zeros and the beginning and the end that's why we 
    use xs and ys on the last equation.
    '''
    
    w,Fw=FFT(t,F)  
    #print(omega.size,Fw.size)
    #Fw = Fw*w/(2j*np.pi)-F[0]*np.exp(2j*w*-0.0229)/2/np.pi+F[-1]*np.exp(2j*w*t[-1])/2/np.pi
    #print(+F[-1]*np.exp(2j*w*t[-1])/2/np.pi)
    #print(F[-1])
    Fw = Fw*w/(2j*np.pi)
    
    #Fw = Fw*omega/(2j*np.pi)#-Ft_ori[0]*np.exp(1j*omega*t_ori[0])/2/np.pi
    
    return w,Fw


 


