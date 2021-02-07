#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 15:05:17 2020

@author: mrgr
"""
import numpy as np
from astropy import units as u
from astropy.constants import GM_sun
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import glob


def PutLabels (x_label, y_label, title):
    
    plt.rcParams["figure.figsize"] = (10,10)
    plt.title(title)
    plt.xlabel(x_label, fontsize=15)
    plt.ylabel(y_label, fontsize=15)
    plt.legend(loc='lower right',prop={'size': 15})
    plt.yticks(size=15)
    plt.xticks(size=15)
    
def Phase(df):
    
    return [float(-1j*np.log(complex(i)/abs(complex(i)))) for i in df.res_adaptive.values]

def Plot(folder, changingvar,namevar,lens_model):
   
    folders_list=sorted(glob.glob(folder))
    print('Files plotted:',folders_list )

    
    ##Amplitude
    
    PutLabels('w','|F|',str(lens_model))
    
    print('Amplitude')
    for fol,c in zip(folders_list, changingvar):
        df=pd.read_csv(fol, sep="\t")
        amp=[float(abs(complex(i))) for i in df.res_adaptive.values]
        w=df.w.values
        plt.plot(w,amp,'-',label=str(namevar)+'='+str(c))
        plt.xscale('log')
        plt.yscale('log')
        #plt.xlim(1, 100)
        #plt.ylim(1, 100)
    #plt.plot(wpoint,pointRefamp, label='point mass', color='k', lw=0.7)
    
    path='./../Results/'+lens_model+'/'
    
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    '''
    
    plt.legend()
    plt.savefig(path + 'Amplitude_diff_'+namevar+'configurations.png')
    plt.show()
    
    
    
    
    ### Phase
    
    PutLabels('w','$\Theta_F$',str(lens_model))
    #create_plots('w','|F|','SIS with core')
    for fol,c in zip(folders_list, changingvar):
        df=pd.read_csv(fol, sep="\t")
        phase=Phase(df)
        w=df.w.values
        plt.plot(w,phase,'-',label=str(namevar)+'='+str(c))
        plt.xscale('log')
        #plt.xlim(1, 100)
        #plt.ylim(1, 100)
         
    #plt.plot(wpoint,pointRefphase, label='point mass', color='k', lw=0.7)
    plt.legend()
    plt.savefig(path + 'Phase_diff_'+namevar+'configurations.png')
    plt.show()
