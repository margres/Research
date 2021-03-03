#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 16:38:02 2021

@author: mrgr
"""
import numpy as np
import matplotlib.pyplot as plt

def PutLayout():
    
    plt.tight_layout()
    plt.rcParams["figure.figsize"] = (8,8)
    params = {'axes.labelsize': 16,
              'axes.titlesize': 16,
              'xtick.labelsize' : 16,
              'ytick.labelsize' : 16,
              'font.size':16,
              'legend.fontsize':12,
              'lines.markersize':6
             }
    plt.rcParams.update(params)
    
def PutLabels (x_label, y_label, title):

    plt.figure()
    plt.title(title)
    plt.xlabel(x_label)#, fontsize=15)
    plt.ylabel(y_label)#, fontsize=15)
    #plt.legend(loc='lower right',prop={'size': 15})
    #plt.yticks(size=15)
    #plt.xticks(size=15)

    PutLayout()

def main():
    
    #amplitude
    
    PutLabels('w','|F|','')
    
    #par it's the changing parameter
    
    ###!!!! idkhow to do this
    for fol,pa in zip(folders_list_point, par):
    
        dfpoint=pd.read_csv(fol, sep="\t")
        amp=dfpoint.Famp.values
        #phase=dfpoint.Fphase.values
    
        plt.plot(w_range,amp,'-',label='y='+str(pa))
        plt.xscale('log')
        #plt.yscale('log')
        plt.legend()
    plt.savefig('./Results/point/amp_analytic_pointmass.png')
    plt.show() 
    
    #Phase 
  
    PutLabels('w','$\Phi_F$','')

    for fol,y in zip(folders_list_point, yLin):
    
        dfpoint=pd.read_csv(fol, sep="\t")
        phase=dfpoint.Fphase.values
        plt.plot(w_range,phase,'-',label='y='+str(y))
        plt.xscale('log')
        #plt.yscale('log')
        plt.legend()
      
    plt.savefig('./Results/point/phase_analytic_pointmass.png') 