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


def GpFunc(xlist, w, y, model_lens):
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

    if model_lens == 'SIS':
        gpx = w * (2.*xlist - 1.) / (1. - xlist)**3.
    
    #if model_lens == 'point':
    #g
    #   g = 0.5*(x1**2.*(1) + x2**2.
   #tau -= np.log(np.sqrt(dx1**2.+dx2**2))

    return gpx

def WFunc(x, w, y, model_lens):
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

    # model-dependant exponential factor
    if model_lens == 'SIS':
        gx = w*(0.5*x**2. - x + y + 0.5)
    
    # cos + i*sin for complex exponential part
    cosv = np.cos(gx)
    sinv = np.sin(gx)
    

    return np.array([j0v*cosv,
                     j0v*sinv,
                     j1v*cosv,
                     j1v*sinv])

def LevinFunc(xmin, xmax, size, w, y, model_lens, cosORsin):
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
    gpx_u_matrix = np.vstack(GpFunc(xlist, w, y, model_lens))*umatrix
    rx_u_matrix = np.vstack(r / (1.-xlist)**2.)*umatrix
    xx_u_matrix = np.vstack(1. / xlist / (1-xlist))*umatrix

    # assign values block by block for L # determined by transposed A matrix
    Lmatrix[:size, :] = np.concatenate((upmatrix, gpx_u_matrix, rx_u_matrix, null_matrix), axis=1)
    Lmatrix[size:2*size, :] = np.concatenate((-1.*gpx_u_matrix, upmatrix, null_matrix, rx_u_matrix), axis=1)
    Lmatrix[2*size:3*size, :] = np.concatenate((-1.*rx_u_matrix, null_matrix, -1.*xx_u_matrix+upmatrix, gpx_u_matrix), axis=1)
    Lmatrix[3*size:, :] = np.concatenate((null_matrix, -1.*rx_u_matrix, -1.*gpx_u_matrix, -1.*xx_u_matrix+upmatrix), axis=1)



    # oscillatory part
    f_osci_min = WFunc(xmin, w, y, model_lens)
    f_osci_max = WFunc(xmax, w, y, model_lens)

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
        return 0
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

    return I
   

def InteFunc(w, y, model_lens='SIS', size=19, accuracy=1e-6, N_step=50, Niter=int(1e5)):
    """
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

            I_test0 = LevinFunc(a, b, size, w, y, model_lens, part_name)
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

                I_test11 = LevinFunc(a, xmid, size, w, y, model_lens, part_name)
                I_test12 = LevinFunc(xmid, b, size, w, y, model_lens, part_name)
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

        print("Running part", part_name)
        print("adaptive subdivision finished with {:} runs".format(Nrun))
        # print("resulted total bounds", len(xbounds_list))
        print("resulted upper bound", b)
        print("flag_succeed", flag_succeed)

    return I_cos_sin


def InteFunc_simple(w, y, model_lens='SIS', size=19):
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
    part_names = ['cos', 'sin']

    I_cos_sin[0] = LevinFunc(xmin, xmax, size, w, y, model_lens, 'cos')
    I_cos_sin[1] = LevinFunc(xmin, xmax, size, w, y, model_lens, 'sin')

    return I_cos_sin


def InteFunc_fix_step(w, y, model_lens='SIS', size=19, N_step=50):
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

    I_cos_sin = np.zeros(2, dtype=float)

    # fix dx for each step 
    xbounds_list = np.linspace(xmin, xmax, N_step)
    for i in range(len(xbounds_list)-1):

        I_test0 = LevinFunc(xbounds_list[i], xbounds_list[i+1], size, w, y, model_lens, 'cos')
        # print("sub-result (cos)", I_test0)
        # accumulate results from the last run
        I_cos_sin[0] += I_test0
    
        I_test0 = LevinFunc(xbounds_list[i], xbounds_list[i+1], size, w, y, model_lens, 'sin')
        # print("sub-result (sin)", I_test0)
        # accumulate results from the last run
        I_cos_sin[1] += I_test0

    return I_cos_sin




if __name__ == '__main__':

    import time
    import cmath 
    
    y = 0.3
    # # results from Mathematica
    # # 2.7270784320701793 + 12.832498219682217*I (integral)
    # # -0.1593982590701816 (phase)

    #const = -1j*w*cmath.exp(1j*w*y**2./2.)
    w_list=[]
    res_simple=[]
    res_fixed=[]
    res_adaptive=[]
    time_simple=[]
    time_fixed=[]
    time_adaptive=[]
    
    
    w_range=np.linspace(0.001,100,1000)

   

    # ++++++++++++++++++++++++++ simple with whole range
    for w in w_range:
        print('W',w)
        const = -1j*w*cmath.exp(1j*w*y**2./2.)
        start = time.time()
        I_cos_sin = InteFunc_simple(w, y)
        print("simple method finished in", time.time()-start)
        print('I_cos', I_cos_sin[0])
        print('I_sin', I_cos_sin[1])
        
        res = const * (I_cos_sin[0] + 1j*I_cos_sin[1])
        res_simple.append(res)
        time_simple.append(time.time()-start)
        print('phase', cmath.phase(res))
    
        # ++++++++++++++++++++++++++ fixed subdivision
        start = time.time()
        I_cos_sin = InteFunc_fix_step(w, y)
        print("fixed subdivision finished in", time.time()-start)
        print('I_cos', I_cos_sin[0])
        print('I_sin', I_cos_sin[1])
    
        res = const * (I_cos_sin[0] + 1j*I_cos_sin[1])
        res_fixed.append(res)
        time_fixed.append(time.time()-start)
        print('phase', cmath.phase(res))
    
        # ++++++++++++++++++++++++++ optimal with adaptive subdivision
        
        start = time.time()
        I_cos_sin = InteFunc(w, y)
        print("adaptive subdivision finished in", time.time()-start)
        print('I_cos', I_cos_sin[0])
        print('I_sin', I_cos_sin[1])
    
        res = const * (I_cos_sin[0] + 1j*I_cos_sin[1])
        res_adaptive.append(res)
        time_adaptive.append(time.time()-start)
        
        print('phase', cmath.phase(res))
       
        

    #df = pd.DataFrame(list(zip(res_simple,time_simple,res_fixed,time_fixed,res_adaptive,time_adaptive)),columns=['res_simple','time_simple','res_fixed','time_fixed','res_adaptive','time_adaptive'] )
    df = pd.DataFrame(list(zip(res_simple,time_simple,res_fixed,time_fixed, res_adaptive,time_adaptive)),columns=['res_simple','time_simple','res_fixed','time_fixed','res_adaptive','time_adaptive'] )

    df.to_csv('final_levin.txt', sep='\t')

# +++++++++++++ Running results (2020-05-04, raam)

# simple method finished in 0.005729198455810547
# I_cos 2.7271102100223077
# I_sin 12.836214489975802
# phase -0.15934175989756247

# fixed subdivision finished in 0.15572452545166016
# I_cos 2.7265931428090022
# I_sin 12.832828610891243
# phase -0.15935684162271987

# Running part cos
# adaptive subdivision finished with 78 runs
# resulted upper bound 0.9999999999813736
# flag_succeed True
# Running part sin
# adaptive subdivision finished with 76 runs
# resulted upper bound 0.9999999999254943
# flag_succeed True
# adaptive subdivision finished in 0.7144765853881836
# I_cos 2.727041869235449
# I_sin 12.83258598501708
# phase -0.15939414233453655
