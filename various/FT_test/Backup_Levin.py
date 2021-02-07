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

#*************** GENERAL METHOD  ***************

def Bes(const,x,n_basis,point,order=1):
    '''
    function that creates the matrix A for a Bessel function
    '''

    A=Matrix(np.zeros([2*n_basis,2*n_basis]))
    Id=Matrix(np.zeros([2*n_basis,2*n_basis]))
    for j in range(n_basis):
        for k_i in range(n_basis):
            #k_=k_i+1
            A[j,k_i]=(order - 1)/x
            Id[j,k_i]=1
            A[j+n_basis,k_i]=const
            A[j,k_i+n_basis]=-const
            A[j+n_basis,k_i+n_basis]=-order/x
            Id[j+n_basis,k_i+n_basis]=1
        k_i=0
    return A, Id


def coordinate_transformation(x,b,f,g,w_oscillating):
    '''
    change of the integration variable, From a 0-inf integral to a 0-1 
    '''
    x_change=x/(1-x)
    x_prime=1/(1-x)**2
    
    if b==np.inf:
        b_new=0.9999
    else:
        b_new=b/(1+b)
    if isinstance(f, Symbol):  
        #print(f)
        f=f.subs({x:x_change})*x_prime
        #print(f)
    if isinstance(g, Symbol):
        g=g.subs({x:x_change})
    w_oscillating=simplify(w_oscillating.subs({x:x_change}))
    
    return x_change,x_prime,b_new,f,g,w_oscillating

def chebyshev_right_open(n):
    
    nodes=[np.cos(((1 - 2*j + 2*n)*np.pi)/(4*n))**2 for j in range(1,n+1)]
    #nodes.append(1)
    #nodes.append(0)
    
    return nodes

def right_IMT(t):
    
    IMT=exp(1-1/(1-t))
    
    return IMT

def levin_general(f,g,const,a,b,w_oscillating,n_basis):
    '''
    Levin method applied.
    The collocation points of the basis need to be equidistant.
    g: exponent of the exponential
    f: x
    const: constants in the bessel function argument
    a,b: limits of the integral
    w_oscillating: osciallting parts of the integral
    n_basis: number of basis
    '''

    start_time = time.time()
    n=n_basis
    n2=2*n_basis
    k= Symbol('k')
    
    #i check the input, to see if g oscillates or not. If it doesn't we put it on the rhs
    
    if isinstance(g, complex):  #if the exponent of the exponential is complex then it oscilates
        A_g=I*g.diff()  
    elif g==0:
        A_g=1  
    else:
        A_g=1
        f=f*exp(g)
        w_oscillating=w_oscillating/exp(g)
    
    #print('before:\n',b,'\n',f,'\n',g,'\n',w_oscillating)
    if b==np.inf:
        x_change,x_Jacobian,b,f,g,w_oscillating=coordinate_transformation(x,b,f,g,w_oscillating)
        point=chebyshev_right_open(n)
        #point=[a+(j-1)*(b-a)/(n-1) for j in range(1,n+1)]
    else:
        x_change=x
        #point=[a+(j-1)*(b-a)/(n-1) for j in range(1,n+1)]
        point=chebyshev_right_open(n)

   
    #print('after:\n', x_change,'\n', x_Jacobian,'\n',b,'\n',f,'\n',g,'\n',w_oscillating)
    #print(w_oscillating,'\n',simplify(f))
    
    d=(a+b)/2+0.0000000001
    
    u=(x-d)**(k-1) #points in the range of integration
    uprime=(k-1)*(x-d)**(k-2) #derivative of u

    rhs = np.zeros(n2)  #right hand side, aka f(x)
    #print(point)
    for i,r in enumerate(point):
        rhs[i]=f.subs({x:r})
    #rhs=Matrix(rhs)
    
    #levin's approximation of the bessel function
    A, Id=Bes(const,x_change,n_basis,point)
        
        
    A_f=A*u*A_g+Id*uprime  #here I create the matrix with the contribute of the Bessel function and the exponential
    

    for j in range(n2):
        if j<n:
            val=point[j]
        else:
            val=point[j-n]
        for k_i in range(n2):
            if k_i<n:
                k_=k_i+1
            else:
                k_=k_i+1 -n
           
            A_f[j,k_i]=A_f[j,k_i].subs({x:val,k:k_})

        k_i=0
    #print(A_f)
    if False:       
        c=symarray('c',n2)
        rhs=rhs.reshape(n2,1)
        A_n=Matrix(np.hstack((A_f,rhs)))      
        coefficients_LU=solve_linear_system_LU(A_n,c)
        
        coefficients=np.zeros(len(c),dtype=np.complex128)
        for i,value in enumerate(coefficients_LU.values()):
            coefficients[i]=expand(value)
            
    if True: #I checked which one is the faster
        c=symarray('c',n2)
        A_f=np.array(A_f,dtype=np.complex128)
        coefficients=np.linalg.solve(A_f,rhs)
   
    monomials_inf=[u.subs({x:a,k:j}) for j in range(1,n+1)]
    monomials_sup=[u.subs({x:b,k:j}) for j in range(1,n+1)]
  
    result=complex(N(np.dot(monomials_sup,coefficients[:n])*w_oscillating.subs({x:b})-np.dot(monomials_inf,coefficients[:n])*w_oscillating.subs({x:a})))
    elapsed_time = time.time() - start_time
    print(result)
    #print('Found the coefficient for the non-rapidly-oscillatory f(x) after:', \
        #  time.strftime("%H:%M:%S", time.gmtime(elapsed_time))+' with '+str(n_basis)+' basis')
    print('time: ',elapsed_time)
    return result,elapsed_time
   

if __name__ == "__main__":
    
    x = Symbol('x')
    f=x
    y_n=1
    w_SIS=0.01
    bess_func_arg=w_SIS*y_n
    J = besselj(0, bess_func_arg*x)
    g=w_SIS*(0.5*x**2-x)
    #g=x
    w_oscillating=J*exp(I*g)

    
    
    basis=[]
    result=[]
    elapsed_time=[]
    range_int=[]
    
    '''
    for s in range(1,5):
        if s==1:
            pass
        else:
            s=s*2
    '''    
    for i in range(4,10):    
        print(i)
        r,elaps_time=levin_general(f,g,bess_func_arg,0.0000001,1,w_oscillating,n_basis=i*5) 
        result.append(r)
        elapsed_time.append(elaps_time)
        basis.append(i*5)
        range_int.append(1)
    i=0    
        
    df = pd.DataFrame(list(zip(basis,result,elapsed_time,range_int)),columns=['basis','result','time','integration range'] )
    #print(df)
    df.to_csv('dataframe_fun_og.txt', sep='\t')
    '''
    
    #print(levin_general(f,g*w,w*1,0.0000001,10,19)) 
    n_basis=19
    result=levin_general(f,g,bess_func_arg,0.000000001,1,w_oscillating,n_basis)[0]
    print('basis',n_basis,'result',result)
    '''
  



