#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 10:22:16 2020

@author: mrgr
"""

import numpy as np
from sympy import Symbol,simplify,expand,diff, Function, solve_linear_system_LU, besselj, I,linear_eq_to_matrix,Matrix,exp,N, symarray
from string import ascii_lowercase
#import sympy.abc as abc  
from scipy.special import jv as scipy_besselj

import time
from mpmath import fp
import pandas as pd

# $\int e^{ig(x)}f(x)dx$
# 
# $\int dx\cdot x J(wxy) \cdot e^{iw[0.5x^2-x+\phi]}dx$
# 
# as demonstrated by Levin, we can find an approximation to a less oscillatory solution by collocation.


#*************** BASIC/SCALAR METHOD  ***************

def differential_eq(F,x,g):
    '''
    input:  F(x), the new not rapidly oscillatory function 
            x, variable in which we need to integrate
            g, exponent of the exponential    
            
    output: differential equation that approximates f(x)
    '''
    return F.diff(x)+I*g.diff(x)*F


            
    
def collocation_method(f,n_basis):
    '''
    input:  f(x) function in the integral
            g(x), exponent of the exponential 
            n_basis, amount of basis funcitons
            
    output: the function F(x) which has been approximated using the collocation method.          
    
    '''   
    
    x = Symbol('x')
    F = Function('F')(x)

    a_list=[] #list of the constants in the inear combination of n basis funcitons
    u=[] #basis functions

    '''
    for i,c in zip(range(1,n_basis+1),ascii_lowercase):
        u.append(x**(i-1)) #monomials
        c = Symbol(str(c))
        a_list.append(c)
    '''

    a_list=coeff_list(n_basis)
    
    #print(u)
    print(a_list)
    F=np.asarray(a_list).dot(np.asarray(u))
    return F, a_list

def check_lim(lim):
    '''
    chech the boundaries of the integral
    '''
    if lim == float("inf") or lim == float("-inf"):
        return int(1e6) 
    else:
        return lim
    

def levin_basic(f,g,lim_inf,lim_sup,n_basis=4):
    '''
    Levin method applied.
    The collocation points of the basis need to be equidistant.
    '''
    start_time = time.time()
    x = Symbol('x')
    lim_inf=check_lim(lim_inf)
    lim_sup=check_lim(lim_sup)
    
    #I have to add the check if function has only x as a variable
  
    x_val=[lim_inf+(j-1)*(lim_sup-lim_inf)/(n_basis-1) for j in range(1,n_basis+1)]
    '''
    for i in range(n_basis):
        t=i/n_basis
        u = 1-t
        x_val.append(lim_inf*u +  lim_sup*t)
        #print(x_val)
    '''    
        
    F,variables=collocation_method(g,f,n_basis)
    new_eq=differential_eq(F,x,g)-f
    print(new_eq)
    
    equations_list=[]
    for i, x_i in enumerate(x_val):
        equations_list.append(simplify(expand(new_eq.subs({x:x_i}))))

    A, b = linear_eq_to_matrix(equations_list, variables)
    elapsed_time = time.time() - start_time
    print('Created set of linear equations after: ',time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    #coefficients= list(fp.lu_solve(A, b), variables).args[0])
    A_n=Matrix(np.hstack((A,b)))
 
    coefficients_LU= solve_linear_system_LU(A_n,variables)
    
    elapsed_time = time.time() - start_time
    print('Found the coefficient for the non-rapidly-oscillatory f(x) after:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    
    coefficients=np.zeros(len(variables),dtype=np.complex128)
    for i,value in enumerate(coefficients_LU.values()):
        coefficients[i]=expand(value)
        #print(sol[i])
        
    #elapsed_time = time.time() - start_time
    F_new=F.copy()
    for coef,coef_val in zip(variables, coefficients):
        F_new=F_new.subs({coef:coef_val})
    evaluate=F_new*exp(I*g)
    solution=N(simplify(evaluate.subs({x:lim_sup})-evaluate.subs({x:lim_inf}))) #integral evaluated at the boundaries
    #print(solution)
    return solution