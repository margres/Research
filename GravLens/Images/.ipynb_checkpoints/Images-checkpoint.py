# -*- coding: utf-8 -*-
# @Author: lshuns
# @Date:   2020-08-03 16:53:12
# @Last Modified by:   lshuns
# @Last Modified time: 2020-08-29 16:08:38

### solve the lens equation

######### Coordinates convention:
############# the origin is set in the centre of perturbation (external shear) and the x-aixs is parallel to the direction of shear, that means gamma2 = 0
############# the source is set in the origin (xs=0)

######### Caveat:
############# images with theta ~ [0,0.5*np.pi,np.pi,1.5*np.pi,2.*np.pi] is ignored

import numpy as np  
import scipy.optimize as op


def TFunc(x12, xL12, lens_model, kappa=0, gamma=0):
    """
    the time-delay function (Fermat potential)
    Parameters
    ----------
    x12: a list of 1-d numpy arrays [x1, x2]
        Light impact position, coordinates in the lens plane.
    xL12: a list of 1-d numpy arrays [xL1, xL2]
        lens center position, coordinates in the lens plane.
    lens_model: str
        Lens model, supported model ('point').
    kappa: float (optional, default=0)
        convergence of external shear.
    gamma: float (optional, default=0)
        shear of external shear.
    """

    # geometrical term (including external shear)
    x1 = x12[0]
    x2 = x12[1]
    tau = 0.5*(x1**2.*(1-kappa-gamma) + x2**2.*(1-kappa+gamma))

    # distance between light impact position and lens position
    dx1 = np.absolute(x1-xL12[0])
    dx2 = np.absolute(x2-xL12[1])

    # deflection potential
    if lens_model == 'point':
        tau -= np.log(np.sqrt(dx1**2.+dx2**2.))
    elif lens_model== 'SIS':
        tau -= np.sqrt(dx1**2.+dx2**2.)
    
    return tau


def dTFunc(x12, xL12, lens_model, kappa=0, gamma=0):
    """
    the first derivative of time-delay function (Fermat potential)
    Parameters
    ----------
    x12: a list of 1-d numpy arrays [x1, x2]
        Light impact position, coordinates in the lens plane.
    xL12: a list of 1-d numpy arrays [xL1, xL2]
        lens center position, coordinates in the lens plane.
    lens_model: str
        Lens model, supported model ('point').
    kappa: float (optional, default=0)
        convergence of external shear.
    gamma: float (optional, default=0)
        shear of external shear.
    """

    # geometrical term (including external shear)
    x1 = x12[0]
    x2 = x12[1]
    dtaudx1 = (1-kappa-gamma)*x1
    dtaudx2 = (1-kappa+gamma)*x2

    # distance between light impact position and lens position
    dx1 = np.absolute(x1-xL12[0])
    dx2 = np.absolute(x2-xL12[1])

    # deflection potential
    if lens_model == 'point'  :
        dx12dx22 = dx1**2.+dx2**2
        dtaudx1 -= dx1/dx12dx22
        dtaudx1 -= dx2/dx12dx22

    elif lens_model == 'SIS':
        dx12dx22 = np.sqrt(dx1**2+dx2**2)
        dtaudx1 -= dx1/dx12dx22
        dtaudx1 -= dx2/dx12dx22
        
        
    return dtaudx1, dtaudx2


def ThetaOrRFunc(theta_t, xL12, lens_model, kappa=0, gamma=0, thetaORr='theta'):
    """
    the theta part or the r part of the lens equation
        Note: dx1 = r_t*cosTheta_t, dx2 = r_t*sinTheta_t
    Parameters
    ----------
    theta_t: 1-d numpy arrays
        Angular coordinate of dx(=xI-xL) in lens plane.
    xL12: a list of 1-d numpy arrays [xL1, xL2]
        Lens center position, coordinates in the lens plane.
    lens_model: str
        Lens model, supported model ('point').
    kappa: float (optional, default=0)
        Convergence of external shear.
    gamma: float (optional, default=0)
        Shear of external shear.
    thetaORr: str (optional, default='theta') 
        Return the theta ('theta') or r ('r') part of the lens equation.
    """

    # lens position
    xL1 = xL12[0]
    xL2 = xL12[1]

    if lens_model=='point':
        # external-shear-related constants
        A1 = (1.-kappa-gamma)
        A2 = (1.-kappa+gamma)
        # theta 
        cosTheta = np.cos(theta_t)
        sinTheta = np.sin(theta_t)
        #
        C = xL1*A1*sinTheta - xL2*A2*cosTheta
        rt = C/(A2-A1)/cosTheta/sinTheta
        #
        if thetaORr == 'theta':
            return A1*xL1 + A1*rt*cosTheta - cosTheta/rt
            # return A2*xL2 + A2*rt*sinTheta - sinTheta/rt
        elif thetaORr == 'r':
            return rt
        else:
            raise Exception('Unsupported thetaORr value! using either r or theta!')
            
    elif lens_model=='SIS':
        cosTheta = np.cos(theta_t)
        sinTheta = np.sin(theta_t)
        secTheta = 1/np.cos(theta_t)
        #rt=1
        #rt=np.sqrt(xL1**2+xL2**2)
        #
        if thetaORr == 'theta':
            return xL1*sinTheta-xL2*cosTheta
        elif thetaORr == 'r':
            print(theta_t)
            try:
                return 1/2*( -2 *xL1*secTheta+ secTheta**2*( -np.sqrt(cosTheta**4 - 4*xL1*cosTheta**3 ))+1)
            except:
                return 1/2*( -2 *xL1*secTheta+ secTheta**2*np.sqrt(cosTheta**4 - 4*xL1*cosTheta**3 )+1)
            
            #rt*cosTheta+xL1-cosTheta*np.sqrt(rt)
        else:
            raise Exception('Unsupported thetaORr value! using either r or theta!')
        


def muFunc(x12, xL12, lens_model, kappa=0, gamma=0):
    """
    the magnification factor
    Parameters
    ----------
    x12: a list of 1-d numpy arrays [x1, x2]
        Light impact position, coordinates in the lens plane.
    xL12: a list of 1-d numpy arrays [xL1, xL2]
        lens center position, coordinates in the lens plane.
    lens_model: str
        Lens model, supported model ('point').
    kappa: float (optional, default=0)
        convergence of external shear.
    gamma: float (optional, default=0)
        shear of external shear.
    """

    # distance between light impact position and lens position
    dx1 = np.absolute(x12[0]-xL12[0])
    dx2 = np.absolute(x12[1]-xL12[1])    

    # second order derivative of deflection potential
    if lens_model == 'point':
        dx22mdx12 = dx2**2.-dx1**2.
        dx12pdx22_2 = (dx1**2.+dx2**2.)**2.
        # d^2psi/dx1^2
        dpsid11 = dx22mdx12/dx12pdx22_2
        # d^2psi/dx2^2
        dpsid22 = -dpsid11
        # d^2psi/dx1dx2
        dpsid12 = -2*dx1*dx2/dx12pdx22_2
        
    elif lens_model == 'SIS':
        dx12pdx22_32 = (dx1**2.+ dx2**2.)**(3/2)
        # d^2psi/dx1^2
        dpsid11 = dx2**2/dx12pdx22_32
        # d^2psi/dx2^2
        dpsid22 = dx1**2/dx12pdx22_32
        # d^2psi/dx1dx2
        dpsid12 = -(dx1*dx2)//dx12pdx22_32
        

    # Jacobian matrix
    j11 = 1. - kappa - gamma - dpsid11
    j22 = 1. - kappa + gamma - dpsid22
    j12 = -dpsid12
    # magnification
    mu = 1./(j11*j22-j12*j12)

    # trace (for image type)
    tr = j11 + j22

    # image type
    flag_min = (mu>0) & (tr>0)
    flag_max = (mu>0) & (tr<0)
    flag_saddle = (mu<0)
    ##
    Itype = np.empty(len(mu), dtype=object)
    Itype[flag_min] = 'min'
    Itype[flag_max] = 'max'
    Itype[flag_saddle] = 'saddle'

    return mu, Itype


def Images(xL12, lens_model, kappa=0, gamma=0, return_mu=False, return_T=False):
    """
    Solving the lens equation
    Parameters
    ----------
    xL12: a list of 1-d numpy arrays [xL1, xL2]
        lens center position, coordinates in the lens plane.
    lens_model: str
        Lens model, supported model ('point').
    kappa: float (optional, default=0)
        convergence of external shear.
    gamma: float (optional, default=0)
        shear of external shear.
    return_mu: bool, (optional, default=False)
        Return the maginification (or not).
    return_T: bool, (optional, default=False)
        Return the time delay (or not).
    """
    
    
    # solve the theta-function
    N_theta_t = 100
    d_theta_t = 1e-3

    node_theta_t = np.array([0, 0.5*np.pi, np.pi, 1.5*np.pi, 2.*np.pi])

    theta_t_res = []

    # +++++++++++++ solve the lens equation
    #if gamma!=0 and kappa!=0:
    
    for i in range(len(node_theta_t)-1):

        theta_t = np.linspace(node_theta_t[i]+d_theta_t, node_theta_t[i+1]-d_theta_t, N_theta_t)
        theta_t_f = ThetaOrRFunc(theta_t, xL12, lens_model, kappa, gamma, 'theta')

        # those with root
        flag_root = (theta_t_f[1:] * theta_t_f[:-1]) <=0
        theta_t_min = theta_t[:-1][flag_root]
        theta_t_max = theta_t[1:][flag_root]
        for j in range(len(theta_t_min)):
            tmp = op.brentq(ThetaOrRFunc, theta_t_min[j],theta_t_max[j], args=(xL12, lens_model, kappa, gamma, 'theta'))
            theta_t_res.append(tmp)

    # corresponding r_t
    theta_t_res = np.array(theta_t_res)
    r_t = ThetaOrRFunc(theta_t_res, xL12, lens_model, kappa, gamma, 'r')
    
    
    # true solutions
    true_flag = r_t>1e-5
    theta_t_res = theta_t_res[true_flag]
    if not isinstance(r_t, int):
        r_t = r_t[true_flag]
    nimages = len(theta_t_res)
    # to x, y
    dx1 = r_t*np.cos(theta_t_res)
    dx2 = r_t*np.sin(theta_t_res)
    xI12 = [dx1 + xL12[0], dx2 + xL12[1]]

    # +++++++++++++ magnification 
    if return_mu:
        mag, Itype = muFunc(xI12, xL12, lens_model, kappa, gamma)
    else:
        mag = None
        Itype = None

    # +++++++++++++ time delay
    if return_T:
        tau = TFunc(xI12, xL12, lens_model, kappa, gamma)
    else:
        tau = None

    return nimages, xI12, mag, tau, Itype

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    # lens
    #lens_model = 'point'
    lens_model= 'SIS'
    xL1 = 0.5
    xL2 = 0.5

    # external shear
    kappa = 0
    gamma = 0

    n_steps=800
    n_bins=800
    xmin=-3
    xmax=3

    x_range=xmax-xmin
    x_lin=np.linspace(xmin,xmax,n_steps)
    y_lin=np.linspace(xmin,xmax,n_steps)

    X,Y = np.meshgrid(x_lin, y_lin) # grid of point
    
    tau = TFunc([X,Y], [xL1, xL2], lens_model, kappa, gamma)
    
    
    nimages, xI12, muI, tauI,Itype = Images([xL1, xL2], lens_model, kappa, gamma, return_mu=False, return_T=True) 
    print('number of images', nimages)
    print('positions', xI12)
    print('magnification', muI)
    print('time delay', tauI)

    #contour plot
    fig = plt.figure(dpi=100)
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height]) 
    # image
    plt.scatter(xI12[0], xI12[1], color='r', s=4)
    # contour
    cp = ax.contour(X, Y, tau, np.linspace(0,1,50), linewidths=0.6, extent=[-2,2,-2,2], colors='black')
    #plt.xlim(-0.15,0.15)
    #plt.ylim(-0.15,0.15)
    plt.gca().set_aspect('equal', adjustable='box')
    cp.ax.set_ylabel('y', fontsize=13)
    cp.ax.set_xlabel('x', fontsize=13)
    
    plt.title('Contour plot of time delay', fontsize=13)
    # plt.savefig('./test/contour_plot_with_points.png')
    plt.show()