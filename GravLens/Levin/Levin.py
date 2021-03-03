 # -*- coding: utf-8 -*-
# @Author: lshuns
# @Date:   2020-05-01 21:13:39
# @Last Modified by:   lshuns
# @Last Modified time: 2020-05-12 09:41:00

### Levin method for solving highly oscillatory one-dimensional integral
###### Target form: int_0^Infinity dx x J0(rx) exp(i G(x))
###### First do variable change (x=x/(1-x)): int_0^1 dx (x/(1-x)**3) J0(r*(x/(1-x))) exp(i G(x/(1-x)))
###### Transfer to: L * c = [x/(1-x)**3, 0, 0, 0] for cos, or [0, x/(1-x)**3, 0, 0] for sin

import numpy as np 
import scipy.special as ss
import pandas as pd
from FindShift import FirstImage
#import mpmath 
import warnings

def ChebyshevTFunc_simple_x(x, size):
    """
    Chebyshev polynomials of the first kind used as basis functions
        return simple list with form: [T0(x), T1(x),...]
    Parameters
    ----------
    x: float
        the variable
    size: positive int 
        length of requested series.
    """
 
    # first two
    T0 = 1.
    T1 = x
    if size==1:
        return np.array([T0])
    if size==2:
        return np.array([T0, T1])

    # generate the rest of series with recurrence relation
    Tlist = np.zeros(size)
    Tlist[:2] = [T0, T1]
    Tn1 = T0
    Tn2 = T1
    for i in range(2, size):
        tmp = 2.*x*Tn2 - Tn1
        Tlist[i] = tmp
        #
        Tn1 = Tn2
        Tn2 = tmp
        
    #print(Tlist)

    return Tlist


def ChebyshevTFunc(xmin, xmax, size):
    """
    Chebyshev polynomials of the first kind used as basis functions
        return matrix with form: [[T0(x1), T1(x1),...], 
                                    [T0(x2), T1(x2),...],
                                    ...,
                                    [T0(xn), T1(xn),...]]
    Parameters
    ----------
    xmin: float
        the lower bound of integrated range
    xmax: float
        the upper bound of integrated range
    size: positive int 
        length of requested series.
    """

    # Chebyshev nodes used as collocation points
    ilist = np.arange(size) + 1
    xlist = np.cos(((1. - 2.*ilist + 2.*size)*np.pi)/(4.*size))**2. * (xmax-xmin) + xmin
 
    # first two
    Tlist0 = np.ones_like(xlist, dtype=float)
    Tlist1 = xlist
    if size==1:
        return xlist, np.vstack([Tlist0,]).transpose()
    if size==2:
        return xlist, np.vstack([Tlist0, Tlist1]).transpose()

    # generate the rest of series with recurrence relation
    Tlist = [Tlist0, Tlist1]
    Tlistn1 = Tlist0
    Tlistn2 = Tlist1
    for i in range(2, size):
        tmp = 2.*xlist*Tlistn2 - Tlistn1
        Tlist.append(tmp)
        #
        Tlistn1 = Tlistn2
        Tlistn2 = tmp
    #print(np.vstack(Tlist).transpose())
    return xlist, np.vstack(Tlist).transpose()


def ChebyshevUFunc(xmin, xmax, size):
    """
    Chebyshev polynomials of the second kind used as the differentiation of basis functions
        return matrix with form: [[U0(x1), U1(x1),...], 
                                    [U0(x2), U1(x2),...],
                                    ...,
                                    [U0(xn), U1(xn),...]]
    Parameters
    ----------
    xmin: float
        the lower bound of integrated range
    xmax: float
        the upper bound of integrated range
    size: positive int 
        length of requested series.
    """

    # Chebyshev nodes used as collocation points
    ilist = np.arange(size) + 1
    xlist = np.cos(((1. - 2.*ilist + 2.*size)*np.pi)/(4.*size))**2. * (xmax-xmin) + xmin

    # first two
    Ulist0 = np.ones_like(xlist, dtype=float)
    Ulist1 = 2.*xlist
    if size==1:
        return xlist, np.vstack([Ulist0,]).transpose()
    if size==2:
        return xlist, np.vstack([Ulist0, Ulist1]).transpose()

    # generate the rest of series with recurrence relation
    Ulist = [Ulist0, Ulist1]
    Ulistn1 = Ulist0
    Ulistn2 = Ulist1
    for i in range(2, size):
        tmp = 2.*xlist*Ulistn2 - Ulistn1
        Ulist.append(tmp)
        #
        Ulistn1 = Ulistn2
        Ulistn2 = tmp

    return xlist, np.vstack(Ulist).transpose()


def GpFunc(xlist, w, y, model_lens,fact):
    """
    the differentiation of the G(x) function shown in exponential factor
    Parameters
    ----------
    xlist: 1-d numpy array
        the integrated variable
    w: float
        dimensionless frequency from lens problem
    y: float
        impact parameter from lens problem
    model_lens: string
        lens model 
    """
    a,b,c,p=fact[0], fact[1], fact[2],fact[3]

    if model_lens == 'SIS':
        gpx = w * (2.*xlist - 1.) / (1. - xlist)**3.
        
    elif model_lens == 'SIScore':
      
        sq=(b**2+xlist**2/(c**2*(xlist - 1)**2))**(0.5)
        gpx = w * (a*xlist - c**2*xlist*sq)/(c**2*(xlist - 1)**3*sq)

    elif model_lens == 'point':
        #gpx=-w*(0.5*xlist**2 - 2*xlist +1)/((1-xlist)**3 *xlist)
        gpx=-w*(1 - 2*xlist)/((1-xlist)**3*xlist)
        
    elif model_lens == 'softenedpowerlaw':
   
        #ppart=(b**2+xlist**2/(c**2*(xlist - 1)**2))**(p/2 - 1)
        gpx = w *  (xlist*(1. - 1.*a*p*(b**2 + xlist**2/(-1 + xlist)**2)**((-2 + p)/2)))/(1 - xlist)**3
    

    elif model_lens == 'softenedpowerlawkappa':
             
        if p!=0:              
            
            gpx=w*(xlist/(1-xlist)**3 -( (a**(2 - p)*b**p)/(p*(xlist- 1)*xlist) - (a**(2 - p)*(xlist/(1 - xlist))**p *((b**2 *(1 - xlist)**2)/xlist**2 + 1)**(p/2))/(p*(xlist - 1)*xlist)) )           
            #print(gpx)
            #gpx=(a**(2 - p)*(b**p - (1 + (b**2*(1 - xlist)**2)/xlist**2)**(p/2)*(xlist/(1 - xlist))**p))/(p *(-1 + xlist)*xlist)
        else:          
            gpx= w*( xlist/(1-xlist)**3 + a**2*np.log(1 + xlist**2/(b**2*(1 - xlist)**2))/((xlist - 1)*xlist))
  
    elif model_lens == 'nfw':
        gpx=[]
        for xx in xlist:
            if xx<1:
                gp=(-xx**2 + 4*(xx**2 - 2*xx + 1*np.log(xx/(2 - 2*xx)) + (4*(xx**2 - 2*xx + 1)*np.arctanh(np.sqrt((1-2*xx)/(xx - 1)**2)))/np.sqrt((1 - 2*xx)/(xx- 1)**2))/((xx - 1)**3*xx))
            else:
                gp=(4*(1 - xx)**2*np.log(xx/(2- 2*xx))-xx**2)/((xx - 1)**3*xx) + (4*xx*np.arcatnh(np.sqrt((2*xx-1)/(xx-1)**2)))/((xx- 1)*np.sqrt((2*xx - 1)/(xx-1)**2)*(xx**2-4*xx+2))
            gpx.append(gp)
        print('gpx',gpx)
    else:
        gpx=0
        raise Exception('Unsupported lens model')
        
    if np.isnan(gpx).any():
        print(gpx)
        raise Exception('Unsupported value in the derivative')
    #print(gpx)
    return gpx

def WFunc(x, w, y, phim, model_lens,fact):
    
    """
    oscillatory part of integrand
        it is extended to a vetor to meet Levin requirement w' = A w
    Parameters
    ----------
    x: float
        the integrated variable
    w: float
        dimensionless frequency from lens problem
    y: float
        impact parameter from lens problem
    model_lens: 
        lens model 
    """

    # variable change
    x = x/(1.-x)

    # Bessel function
    j0v = ss.j0(w*y*x)
    j1v = ss.j1(w*y*x)
    
    #print(j0v)
    
    #print(j1v)
    
    
    a,b,c,p =fact[0],fact[1],fact[2],fact[3]
    # model-dependant exponential factor
    if model_lens == 'SIS':
        pot=x
        
    elif model_lens== 'SIScore':
        
        pot=a*(x**2 + b**2)**(0.5)
  
    elif model_lens == 'softenedpowerlaw':
        
        if 0<p<2:
        
            pot=a*(x**2+b**2)**(p/2) - a*b**p
        
        else: 
             Exception('Unsupported p value')

        
    elif model_lens == 'softenedpowerlawkappa':
        
        #isothermal power law for p=0
        #modified Hubble model for p= 0
        #Plummer model for p =-2
        
        
        if p>0 and b==0:      
            #eq 28
            pot=1/p**2 * a**(2-p) *x**p
               
        elif p<1 and b!=0 and p!=0:
            
            '''
            if p==1:
                #isothermal ellipsoid
                ph=a/x*(np.sqrt(x**2+b**2)-x)
                pot=x*ph - a*b*np.log((b+np.sqrt(b**2+x**2))/(2*a))
             '''       
             
            t1= 1/p**2 * a**(2-p)*(x+1e-5)**p *ss.hyp2f1(-p/2, -p/2, 1-p/2, -b**2/(x+1e-5)**2)
            t2= 1/p*a**(2-p)*b**p*np.log((x+1e-5)/b)
            t3= 1/(2*p) * a**(2-p)*b**p*(np.euler_gamma-ss.digamma(-p/2))            
            pot=t1 - t2 - t3
            
            '''     
            elif p==0 and b!=0:
                ##check again this
                pot=-1/2 * a**2* mpmath.fp.polylog(2,x**2/b**2)
                #print('pot',mpmath.fp.polylog(2,x**2/b**2))
            '''    
        else:
        #if (b==0 and p<0) or p>=2 or p=0:
            #we can not have b=0 and p<0
            raise Exception('Unsupported p value')
            

    elif model_lens == 'point':
        pot=np.log(x+1e-6)
        
    elif model_lens=='nfw':
        #k=1
        if x<1:
            pot= 2*(np.log(x/2)**2-np.arctanh(np.sqrt(1-x**2))**2)
        else:
            pot= 2*(np.log(x/2)**2+np.arctanh(np.sqrt(-1+x**2))**2)
            
    
    gx= w*(0.5*x**2. - pot + phim)
  
    # cos + i*sin for complex exponential part
    cosv = np.cos(float(gx))
    sinv = np.sin(float(gx))
    
    #print(gx)
    
    if np.isnan(float(gx)).any():
        print(gx)
        raise Exception('Unsupported value in the exponential')
    

    return np.array([j0v*cosv,
                     j0v*sinv,
                     j1v*cosv,
                     j1v*sinv])

def LevinFunc(xmin, xmax, size, w, y, phim, model_lens, cosORsin,fact):
    """
    Levin method to solve the integral with range (xmin, xmax)
        Using the linear equations
            L * c = rhs
            where 
                L = I * u' + A * u
                the form of matrix A is hard-coded, given our integrand feature from axisymmetric lens problem
            where
                rhs = [f(x), 0, 0, 0] for cos
                rhs = [0, f(x), 0, 0] for sin
    Parameters
    ----------
    xmin: float
        the lower bound of integrated range
    xmax: float
        the upper bound of integrated range
    size: positive int 
        length of requested series.
    w: float
        dimensionless frequency from lens problem
    y: float
        impact parameter from lens problem
    model_lens: string
        lens model 
        
    cosORsin: string ('cos', 'sin')
        cos part or sin part 
    """

    # the basis functions
    xlist, umatrix = ChebyshevTFunc(xmin, xmax, size)
    # differentiation of the basis functions
    if size == 1:
        upmatrix = np.vstack([np.zeros(size),])
    else:
        upmatrix = np.zeros((size, size), dtype=float)
        upmatrix[:,1:] = ChebyshevUFunc(xmin, xmax, size)[1][:,:-1] # first column is zero
        upmatrix *= np.array([np.arange(size),]*size)

    # factor within Bessel function
    r = w*y
    
    # hard-coded dimension of A matrix
    m = 4 
    Lmatrix = np.zeros((m*size, m*size), dtype=float)
    # components for A * u matrix
    #   variable change is already taken into account
    null_matrix = np.zeros((size, size))
    gpx_u_matrix = np.vstack(GpFunc(xlist, w, y, model_lens,fact))*umatrix
    rx_u_matrix = np.vstack(r / (1.-xlist)**2.)*umatrix
    xx_u_matrix = np.vstack(1. / xlist / (1-xlist))*umatrix

    # assign values block by block for L # determined by transposed A matrix
    Lmatrix[:size, :] = np.concatenate((upmatrix, gpx_u_matrix, rx_u_matrix, null_matrix), axis=1)
    Lmatrix[size:2*size, :] = np.concatenate((-1.*gpx_u_matrix, upmatrix, null_matrix, rx_u_matrix), axis=1)
    Lmatrix[2*size:3*size, :] = np.concatenate((-1.*rx_u_matrix, null_matrix, -1.*xx_u_matrix+upmatrix, gpx_u_matrix), axis=1)
    Lmatrix[3*size:, :] = np.concatenate((null_matrix, -1.*rx_u_matrix, -1.*gpx_u_matrix, -1.*xx_u_matrix+upmatrix), axis=1)



    # oscillatory part
    f_osci_min = WFunc(xmin, w, y,phim, model_lens, fact)
    f_osci_max = WFunc(xmax, w, y,phim, model_lens, fact)

    # basis functions
    ulist_min = ChebyshevTFunc_simple_x(xmin, size)
    ulist_max = ChebyshevTFunc_simple_x(xmax, size)


    # right side of linear equations f(x) = (x/(1-x)**3) after variable change    
    rhslist = np.zeros(m*size, dtype=float)
    if cosORsin == 'cos':
    #   [f(x), 0, 0, 0] for cos
        rhslist[:size] = xlist/(1.-xlist)**3.
    elif cosORsin == 'sin':
    #   [0, f(x), 0, 0] for sin    
        rhslist[size:2*size] = xlist/(1.-xlist)**3.
    else:
        raise Exception("Unsupported cosORsin value !")

    # solve linear equations
    try:
        clist = np.linalg.solve(Lmatrix, rhslist)
    except np.linalg.LinAlgError:
        #raise Exception("LinAlgError")
        return  0
    # print('clist', clist)

    # collocation approximation
    p_min = np.array([  np.dot(clist[:size], ulist_min),
                        np.dot(clist[size:2*size], ulist_min),
                        np.dot(clist[2*size:3*size], ulist_min),
                        np.dot(clist[3*size:], ulist_min)])
    p_max = np.array([  np.dot(clist[:size], ulist_max),
                        np.dot(clist[size:2*size], ulist_max),
                        np.dot(clist[2*size:3*size], ulist_max),
                        np.dot(clist[3*size:], ulist_max)])
    
    # integral results
    I = np.dot(p_max, f_osci_max) - np.dot(p_min, f_osci_min)
    #print('Imax',np.dot(p_max, f_osci_max) )
    #print('Imin',np.dot(p_min, f_osci_min))
    
    

    return I
   

def InteFunc(w, y, model_lens='SIScore',fact=[1,0,1], size=19, accuracy=1e-6, N_step=50, Niter=int(1e5)):
    """
    
    accuracy= 1e-6
    
    Solve the integral
        int_0^1 dx (x/(1-x)**3) J0(r*(x/(1-x))) exp(i G(x/(1-x)))
        Using
            Levin method + adaptive subdivision
    Parameters
    ----------
    w: float
        dimensionless frequency from lens problem
    y: float
        impact parameter from lens problem
    model_lens: string ('SIS'(default), ...)
        lens model 
    size: positive int (default=19)
        length of requested series.
    
    accuracy: float (default=1e-5)
        tolerable accuracy
    N_step: int (default=50)
        first guess of steps for subdivision
    Niter: int (default=int(1e5))
        the maximum iterating running
    """

    # +++ adaptive subdivision until meet the required accuracy +++ #
    # hard-coded integral range
    #   by variable change, the whole range is always (0,1)
    xmin = 0.
    xmax = 1.

    I_cos_sin = np.zeros(2, dtype=float)
    part_names = ['cos', 'sin']
    
    phim= FirstImage(y,fact,lens_model)
    
    
        
    # cos and sin parts should be divided separately 
    for i_name in range(len(part_names)):
        part_name = part_names[i_name]
        I_final = 0.
        # xbounds_list = []
        flag_succeed = False

        # fix dx for each step 
        dx = (xmax-xmin)/N_step
        a = xmin
        b = a + dx
        
        
        for Nrun in range(Niter):

            # avoid surpassing the upper bound
            if b >= xmax:
                # if this is the case, refine the binning
                dx = (xmax-a)/2.
                # check if dx meets the accuracy
                if dx < accuracy:
                    flag_succeed = True
                    break
                b = a + dx
            
            I_test0 = LevinFunc(a, b, size, w, y,phim, model_lens, part_name,fact)
            # # break with LinAlgError
            # if I_test0==0:
            #     flag_succeed = (b-a)
            #     print("Here 0")
            #     break
            # print("new sub-result", I_test0)

            # avoid accuracy check for the first run
            if I_final != 0.:
                # define the whole accuracy by comparing the new sub-result with the whole result
                diff = np.absolute(I_test0/I_final)
    
                
                # print("I_final", I_final)
                # print("diff", diff)
                if diff < accuracy:
                    flag_succeed = True
                    break

            # keep cutting toward left until it meets the accuracy
            diff_left = 1.
            flag_left_succeed = False
            for Nrun_left in range(Niter):
            
                # split to half
                xmid = (b+a)/2.
                # check if dx meets the accuracy
                if (b-xmid) < accuracy:
                    flag_succeed = True
                    break

                I_test11 = LevinFunc(a, xmid, size, w, y,phim, model_lens, part_name,fact)
                I_test12 = LevinFunc(xmid, b, size, w, y,phim, model_lens, part_name,fact)
                # # break with LinAlgError
                # if (I_test11==0) or (I_test12==0):
                #     flag_succeed = (b-a)
                #     print("here 11")
                #     break

                I_test1 = I_test11 + I_test12
                # define the middle-way accuracy by looking at the difference between half-split result and original result
                diff_left = np.absolute(I_test1-I_test0)
                if diff_left < accuracy:
                    flag_left_succeed = True
                    break

                # iterating
                b = xmid
                I_test0 = I_test11

            # accumulate results from the last run
            I_final += I_test0
            # save bound from the last run
            # xbounds_list.append(b)
            # print("left iterating finished with {:} runs".format(Nrun_left))
            # print("resulted upper bound", b)

            ### move forward to right side
            a = b 
            b = a + dx

        I_cos_sin[i_name] = I_final

        #print("Running part", part_name)
        #print("adaptive subdivision finished with {:} runs".format(Nrun))
        # print("resulted total bounds", len(xbounds_list))
        #print("resulted upper bound", b)
        #print("flag_succeed", flag_succeed)

    return I_cos_sin


def InteFunc_simple(w, y, model_lens='SIScore', fact=[1,0,1],size=19):
    """
    Solve the integral
        int_0^1 dx (x/(1-x)**3) J0(r*(x/(1-x))) exp(i G(x/(1-x)))
        Using
            Levin method 
            without any optimization
    Parameters
    ----------
    w: float
        dimensionless frequency from lens problem
    y: float
        impact parameter from lens problem
    model_lens: string ('SIS'(default), ...)
        lens model 
    size: positive int (default=19)
        length of requested series.
    """


    # hard-coded integral range
    #   by variable change, the whole range is always (0,1)
    xmin = 0.
    xmax = 1.-1e-9

    I_cos_sin = np.zeros(2, dtype=float)
    #part_names = ['cos', 'sin']

    I_cos_sin[0] = LevinFunc(xmin, xmax, size, w, y, phim, model_lens, 'cos',fact)
    I_cos_sin[1] = LevinFunc(xmin, xmax, size, w, y, phim,  model_lens, 'sin',fact)

    return I_cos_sin


def InteFunc_fix_step(w, y, model_lens='SIScore',fact=[1,0,1], size=19, N_step=50):
    """
    Solve the integral
        int_0^1 dx (x/(1-x)**3) J0(r*(x/(1-x))) exp(i G(x/(1-x)))
        Using
            Levin method + fixed subdivision
    Parameters
    ----------
    w: float
        dimensionless frequency from lens problem
    y: float
        impact parameter from lens problem
    model_lens: string ('SIS'(default), ...)
        lens model 
    size: positive int (default=19)
        length of requested series.
    
    N_step: int (default=50)
        fixed number of steps for subdivision
    """

    # hard-coded integral range
    #   by variable change, the whole range is always (0,1)
    xmin = 0.
    xmax = 1.-1e-9
    phim = FirstImage(y,fact,lens_model)
    
    I_cos_sin = np.zeros(2, dtype=float)

    # fix dx for each step 
    xbounds_list = np.linspace(xmin, xmax, N_step)
    for i in range(len(xbounds_list)-1):

        I_test0 = LevinFunc(xbounds_list[i], xbounds_list[i+1], size, w, y, phim, model_lens, 'cos',fact)
        # print("sub-result (cos)", I_test0)
        # accumulate results from the last run
        I_cos_sin[0] += I_test0
    
        I_test0 = LevinFunc(xbounds_list[i], xbounds_list[i+1], size, w, y, phim, model_lens, 'sin',fact)
        # print("sub-result (sin)", I_test0)
        # accumulate results from the last run
        I_cos_sin[1] += I_test0

    return I_cos_sin

def Smoothing(amp,w,t):
        
    w=np.ndarray.tolist(w)
    flag=[]
    for k in range(len(w)-1):
        if abs(amp[k]-amp[k-1]) > 5. and  abs(amp[k]-amp[k+1])>5:
            flag.append(k)
    try:
        foo = flag
        warnings.warn("This integration may contain some scattered points. Not confident accuracy.")
    except NameError:
        pass 
        
    #return amp,w



def Filename(lens_model):
    
    if lens_model=='powerlaw':
            add_info=lens_model+'_lens_dist_'+str(y)+'_p_'+str(p)+'.txt'
        
    elif lens_model=='softenedpowerlaw':
        
        add_info=lens_model+'_lens_dist_'+str(y)+'_a_'+str(a)+'_b_'+str(b)+ '_p_'+str(p) + '.txt'        
    
    elif lens_model=='softenedpowerlawkappa':
        
        add_info=lens_model+'_lens_dist_'+str(y)+'_a_'+str(a)+'_b_'+str(b) + '_p_'+str(p) + '.txt'  
    
    elif lens_model=='point':
        
        add_info=lens_model+'_lens_dist_'+str(y) + '.txt'  
        
    else:
        add_info=lens_model+'_lens_dist_'+str(y)+'_a_'+str(a)+'_b_'+str(b) + '.txt'  
        
    path='../1D_Results/'+lens_model+'/'+ add_info
    
    if not os.path.exists(path):
        os.makedirs(path)

    
    return path

def LevinMethod(w_range,y, lens_model, fact, typesub='Adaptive'):
    
    a,b,c,p =fact[0],fact[1],fact[2],fact[3]
    
    res=[]
    time=[]
   
    for w in w_range:
        
        print('W',w)
        const = -1j*w*cmath.exp(1j*w*y**2./2.)
    
        
        # ++++++++++++++++++++++++++ optimal with adaptive subdivision
        
        start = time.time()
        if typesub=='Fixed':
            I_cos_sin = InteFunc_fix_step(w, y,lens_model, [a,b,c,p])
        elif typesub=='Adaptive':
            I_cos_sin = InteFunc(w, y,lens_model, [a,b,c,p])
        else:
            raise Exception('Unsupported subdivision type')
        
        #print("adaptive subdivision finished in", time.time()-start)
        #print('I_cos', I_cos_sin[0])
        #print('I_sin', I_cos_sin[1])
    
        restemp = const * (I_cos_sin[0] + 1j*I_cos_sin[1])
        res.append(restemp)  
        time.append(time.time()-start)
        
        #print(abs(restemp))
        #print('phase', cmath.phase(res))
            
        #I take out some points to take into account edge effects  
        res,time,w_range=res[:-50],time[:-50],w_range[:-50]  
        Smoothing(res,w_range,time)
       
        #save
        df = pd.DataFrame(list(zip(w_range,res,time)),columns=['w','result','time'] )        
        path = Filename(lens_model)               
        df.to_csv(path, sep='\t')
     
        

if __name__ == '__main__':

    import time
    import cmath 
    import os
    import sys
    sys.path.append('../')
    from Images.CritCaus import PlotCurves
    
    
    yL1=0.3
    yL2=0 
    y =round((yL1**2+yL2**2)**(0.5),3)

    w_range=np.round(np.linspace(0.001,100,1050),5)
        
    a=1 #amplitude parameter
    b=0.5 #[0,0.25,0.5,0.75,1,1.5]#core
    c=1 #flattening parameter
    p=1 #poewer law value
    fact=[a,b,c,p]

    lens_model='softenedpowerlaw'
    models=['point','SIS','SIScore','softenedpowerlaw','softenedpowerlawkappa']
    
    LevinMethod(w_range,y, lens_model, fact)
    #PlotCurves((0,0),(0,y),0,0,lens_model,[a,b,c,p])
    
    