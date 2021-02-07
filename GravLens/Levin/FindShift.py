#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 11:11:15 2020

@author: mrgr
"""

import numpy as np 
import scipy.optimize as op
import scipy.special as ss



def FindCrit(y,lens_model,fact):
    
    rmaxlist=[5,20,50,100,1000]

    for rmax in rmaxlist:
        N= 100
        rmin = 0.001
        #rmax = 50.
        xt = np.linspace(rmin,rmax,N)
        xt_res=[]
        
        for i in range(N-1):
            if eval(lens_model)(xt[i],y,fact)* eval(lens_model)(xt[i+1],y,fact)<=0:
                tmp=op.brentq(eval(lens_model), xt[i],xt[i+1], args=(y,fact))
                xt_res.append(tmp)
                #print('x minimum: ',tmp)
                #break
                #try 

        try:
            foo = tmp
            print('x image' ,xt_res)
            break
        except NameError:
            xt_res.append(0)
            if rmax==rmaxlist[-1]:
                raise Exception('Could not find the value of the first image')

    return xt_res


def SIScore (x,y,fact):
    
    a=fact[0]
    b=fact[1]

    #derivative of the 1D potential
    phip= a*x/(np.sqrt(b**2+x**2))
    #derivative of the time delay
    tp= (x - y) - phip
    
    return tp

def point (x,y,fact):    
    
    #derivative of the 1D potential
    phip= 1./x
    #derivative of the time delay
    tp= (x - y) - phip
    
    return tp


def softenedpowerlaw(x,y,fact):
    
    a,b,p=fact[0], fact[1], fact[3]
    
    phip= a*p*x*(b**2. + x**2.)**(p/2. - 1.) 
    
    tp= (x - y) - phip
    
    return tp

def softenedpowerlawkappa(x,y,fact):
    
    a,b,p=fact[0], fact[1],fact[3]
    
    if p==0 and b!=0:        
        phip=a**2./x * np.log(1.+ x**2./b**2.)           
    elif p<4:         
        phip=a**(2.-p)/(p*x) * (x**p*(1.+b**2./x**2.)**(p/2.) - b**p)
        #phip= a**(2 - p)/(p*x)  * ((b**2+x**2)**p/2 - b**p)
    else:
        raise Exception('Unsupported lens model')
    
    tp= (x - y) - phip
    
    return tp


def nfw(x,y,fact):
    
   # k=1

    if x>1:
        phip= 2*(2*np.log(x/2))/x - (2*x*np.arctanh(np.sqrt(x**2 - 1)))/((x**2 - 2)*np.sqrt(x**2 - 1))
    else:
        phip= 2*(2*np.arctanh(np.sqrt(1 - x**2)))/(x*np.sqrt(1 - x**2)) + (2*np.log(x/2))/x  
    
    tp= (x - y) - phip

    return tp


def TimeDelay(x,y,fact,lens_model):
    
    a,b,c,p=fact[0], fact[1], fact[2],fact[3]
    
    if x==0:
        phi=0
        
    elif lens_model == 'SIScore':
        phi =  a * np.sqrt(b**2+x**2) #SIS
    elif lens_model == 'point':
        phi = np.log(x)
  
    elif lens_model == 'softenedpowerlaw':
        phi=a*(x**2/c**2+b**2)**(p/2) - a*b**p
        
    elif lens_model == 'softenedpowerlawkappa':        
     
        if p>0 and b==0:            
            phi= 1/p**2 * a**(2-p) *x**p
                 
        elif b!=0 and p!=0:
            if x==0:
                t1=0
            else:
                t1= 1./p**2. * a**(2.-p)*x**p *ss.hyp2f1(-p/2., -p/2., 1.-p/2., -b**2./x**2.)

            #print(ss.hyp2f1(-p/2, -p/2, 1-p/2, -b**2/x**2))
            t2= 1./p*a**(2.-p)*b**p*np.log(x/b)
            t3= 1./(2.*p) * a**(2.-p)*b**p*(np.euler_gamma-ss.digamma(-p/2.))   
            phi= t1 - t2 - t3
            
       # elif p==0 and b!=0:
       #     phi=-1/2 * a**2* mpmath.polylog(2,x**2./b**2.)
      
    elif lens_model == 'nfw':
        #k=1
        
        if x>1:
            #print('sqrt',np.sqrt(-1+x**2))
            #print(np.arctanh(1))
            phi= 2*(np.log(x/2)**2+(np.arctanh(np.sqrt(-1+x**2)))**2)
            #print(phi)
        else:
            phi= 2*(np.log(x/2)**2-(np.arctanh(np.sqrt(1-x**2)))**2)
               
    else:
        raise Exception("Unsupported lens model !")
 
    psi_m = -(0.5*(abs(x-y))**2. - phi)
    
    return psi_m

def FirstImage(y,fact,lens_model):
    
    '''
    We have to scale everything in respect to the first image.
    If we have one more value of x at which we have images we 
    need to realize which is the one related to the first 
    (in the time domain) image.
    '''
    
    xlist=FindCrit(y,lens_model,fact)
    tlist=[]
    try: 
        #if there is more than one image
        for x in xlist:
            t=TimeDelay(x,y,fact,lens_model)
            if np.isnan(t)==False:
                print('t',t)
                tlist.append (t)
        t=np.min(tlist)
        print('tlist',tlist)
    except:
        #only one image
        t=TimeDelay(x,y,fact,lens_model)
    print('phi_m:',t)   
    return t



if __name__ == '__main__':
    
    a=1
    b=0.5
    c=1
    p=1.8
    fact=[a,b,c,p]
    y=np.sqrt(2*0.1**2)
    lens_model='point'
    
    
    t = FirstImage(y,fact,lens_model)
    print('phi_m:',t)