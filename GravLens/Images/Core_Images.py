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


def to_cartesian(r,theta):
    
    '''
    function to change the coordinates from polar to cartesian
    '''   
    x=r*np.cos(theta)
    y=r*np.sin(theta)
    return (x,y)



def TFunc(x12, xL12, lens_model, kappa=0, gamma=0,core=[1,0]):
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
    elif lens_model== 'SIS_core':
        a=core[0]
        b=core[1]
        tau -= a*np.sqrt(dx1**2.+dx2**2.+b**2)
        
    return tau


def dTFunc(x12, xL12, lens_model, kappa=0, gamma=0,core=[1,0]):
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
        dtaudx2 -= dx2/dx12dx22

    elif lens_model == 'SIS':
        dx12dx22 = np.sqrt(dx1**2.+dx2**2.)
        dtaudx1 -= dx1/dx12dx22
        dtaudx2 -= dx2/dx12dx22
                
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
    if thetaORr == 'theta' :
        if lens_model=='point':
            return A1*xL1 + A1*rt*cosTheta - cosTheta/rt
                # return A2*xL2 + A2*rt*sinTheta - sinTheta/rt
        elif lens_model=='SIS':
            return A1*xL1 + A1*rt*cosTheta - cosTheta
                
    elif thetaORr == 'r':            
            return rt
    else:            
        raise Exception('Unsupported thetaORr value! using either r or theta!')
            
  
        
def CriticalThetaOrRFunc(theta_t, lens_model, kappa=0, gamma=0, thetaORr='theta'):
    
     # theta 
    cos2Theta = np.cos(2*theta_t)
    sin2Theta = np.sin(2*theta_t)
    
    # external-shear-related constants
    A = (1 - 2*kappa+kappa**2-2*kappa*gamma-gamma**2)
    B = (2*gamma*cos2Theta- sin2Theta**2)
   

    rt= np.sqrt(-B + np.sqrt(B**2 + 4*A*cos2Theta**2))
    
    if thetaORr == 'theta' :
        if lens_model=='point':
            return A*rt**4+B*rt**2-cos2Theta**2
              
        #elif lens_model=='SIS':
        #    return A1*xL1 + A1*rt*cosTheta - cosTheta
                
    elif thetaORr == 'r':            
            return rt
    
    
def muFunc(x12, xL12, lens_model, kappa=0, gamma=0, FindCrit=False):
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
        dpsid12 = -(dx1*dx2)/dx12pdx22_32
        

    # Jacobian matrix
    j11 = 1. - kappa - gamma - dpsid11
    j22 = 1. - kappa + gamma - dpsid22
    j12 = -dpsid12
    detA=j11*j22-j12*j12
    
    if FindCrit==True:
        
        return detA
    
    # magnification
    mu = 1./detA

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


def CritCaus(r_t, xI12,xL12,kappa,lens_model):
    
    theta=np.linspace(-2*np.pi,2*np.pi,100)
    fig = plt.figure(dpi=100)
    
    if gamma==0:
        
        xyCrit=np.zeros((2,theta.size))
        xyCaus=np.zeros((2,theta.size))

        for i,th in enumerate(theta):
            #the circle is centered where the lens is
            crit=to_cartesian(r_t,th)
            xyCrit[0,i]=crit[0]+xL12[0]
            xyCrit[1,i]=crit[1]+xL12[1]
            caus=to_cartesian(LensEq(r_t,gamma,lens_model, xL12),th)
            xyCaus[0,i]=caus[0]+xL12[0]
            xyCaus[1,i]=caus[1]+xL12[1]
            
        plt.plot(xyCrit[0,:],xyCrit[1,:], '-', c='k', label='critical curves',linewidth=0.7)
        plt.scatter(xyCaus[0,:],xyCaus[1,:], c='k',label='caustic')
        
    else:
        n_steps=800
        xmin=-5
        xmax=5
        
        x_range=xmax-xmin
        x_lin=np.linspace(xmin,xmax,n_steps)
        y_lin=np.linspace(xmin,xmax,n_steps)
        X,Y = np.meshgrid(x_lin, y_lin)
        crit_curv=muFunc((X,Y), xL12, lens_model, kappa, gamma, FindCrit=True)
        cp = plt.contour(X, Y, crit_curv,[0], colors='k',linestyles= '-',label='critical curves', linewidths=0.7) 
        
        #I get the coordinates of the contour plot
        p_1 = cp.collections[0].get_paths()[0]  
        coor_p_1 = p_1.vertices
        xCrit = coor_p_1[:,0]
        yCrit = coor_p_1[:,1]
        #print(xCrit[:10])
        plt.plot(xCrit,yCrit)
        xyCaus=LensEq(np.array((xCrit,yCrit)),gamma,lens_model,xL12)
        #plt.plot(xyCaus[0]+xL12[0],xyCaus[1]+xL12[1], '-', c='k', label='caustics',linewidth=0.7)
        plt.plot(xyCaus[0],xyCaus[1], '-', c='k', label='caustics',linewidth=0.7)
        

    plt.scatter(xI12[0], xI12[1], marker='o',color='r', label='images')
    plt.scatter(xL12[0], xL12[1], marker='x',color='r', label='lens')
    
    plt.ylabel('y', fontsize=13)
    plt.xlabel('x', fontsize=13)
    plt.xlim(-1.5-xL12[0],1.5+xL12[0])
    plt.ylim(-1.5-xL12[1],1.5+xL12[1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc='lower left',frameon=False,prop={'size':10})
    plt.show()
    
def LensEq(r_t,gamma, lens_model,xL12):
    
    if gamma!=0:
        # distance between light impact position and lens position
        dx1 = r_t[0]
        dx2 = r_t[1]
        #dx1 = np.absolute(r_t[0]-xL12[0])
        #dx2 = np.absolute(r_t[1]-xL12[1]) 
       
    if lens_model=='SIS' and gamma==0:
        #for axisymmetric lenses
    
        alpha = np.sign(r_t)
        
    elif lens_model=='point' and gamma==0:
        E_r=1
        alpha= E_r**2/r_t
        
    elif lens_model=='SIS' and gamma!=0:
        
        dx12dx22 = np.sqrt(dx1**2+dx2**2)        
        dx1 = dx1/dx12dx22
        dx2 = dx2/dx12dx22
        
        alpha=np.array((dx1,dx2))
    
    if lens_model == 'point' and gamma!=0:
        '''
        dx12dx22 = dx1**2.+dx2**2
        dx1 = dx1/dx12dx22
        dx2 = dx2/dx12dx22
        '''
        dx12dx22 = np.sqrt(dx1**2+dx2**2)
        dx1= 1/dx12dx22
        dx2= 1/dx12dx22
        alpha=np.array((dx1,dx2))
    
    return r_t - alpha
    
    
    

def Images(xL12, lens_model, kappa=0, gamma=0, b,q,return_mu=True, return_T=False):
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
    
    # +++++++++++++ solve the lens equation
    # 1D problem
    if gamma==0:
        ## in the same line as the lens
        theta = np.arctan(xL12[1]/xL12[0]) # returns [0, pi/2]
        rL = (xL12[0]**2 + xL12[1]**2)**0.5
        
        E_r=1
        
        ## solve r
        if lens_model == 'point':
            ## two solutions
            r_t = np.array([0.5 * (-rL + (rL**2 + 4./(1-kappa))**0.5), 
                            0.5 * (-rL - (rL**2 + 4./(1-kappa))**0.5)])
            
            nimages = 2
            ## to x, y
            dx1 = r_t*np.cos(theta)
            dx2 = r_t*np.sin(theta)
            xI12 = [dx1 + xL12[0], dx2 + xL12[1]]
            
        elif lens_model == 'SIS':
            r_t = np.array([1/(1-kappa),-1/(1-kappa)])                
            nimages = 2
            ## to x, y
            dx1 = r_t*np.cos(theta)
            dx2 = r_t*np.sin(theta)
            xI12 = [dx1, dx2]
            
     
        CritCaus(E_r, xI12,xL12,kappa,lens_model) #wihtout external shear critical curve is the Einstein radius
            
            
    # 2D problem
    else:
       
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
        
        CritCaus(1, xI12,xL12,kappa,lens_model)

    # +++++++++++++ magnification of the images
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
    lens_model= 'point'
    xL1 = 0.5
    xL2 = 0.1
    
    a=1 #lens amplitude
    b=2 #core SIS
    core=[a,b]
    
    # external shear
    kappa = 0.1
    gamma = 0.1

    n_steps=800
    n_bins=800
    xmin=-3
    xmax=3

    x_range=xmax-xmin
    x_lin=np.linspace(xmin,xmax,n_steps)
    y_lin=np.linspace(xmin,xmax,n_steps)

    X,Y = np.meshgrid(x_lin, y_lin) # grid of point
    
    tau = TFunc([X,Y], [xL1, xL2], lens_model, kappa, gamma)
    
    
    nimages, xI12, muI, tauI,Itype = Images([xL1, xL2], lens_model, kappa, gamma, return_mu=True, return_T=True) 
    print('number of images', nimages)
    print('positions', xI12)
    print('magnification', muI)
    print('time delay', tauI)

    #contour plot
    fig = plt.figure(dpi=100)
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height]) 
    # image
    plt.scatter(xI12[0], xI12[1], color='r', s=4, label='images')
    # contour
    cp = ax.contour(X, Y, tau, np.linspace(0,1,50), linewidths=0.6, extent=[-2,2,-2,2], colors='black')
    plt.xlim(-np.max(X)-xL1,np.max(X)+xL1)
    plt.ylim(-np.max(X)-xL2,np.max(X)+xL2)

    plt.scatter(xL1, xL2, marker='x',color='r', label='lens')
    plt.gca().set_aspect('equal', adjustable='box')
    cp.ax.set_ylabel('y', fontsize=13)
    cp.ax.set_xlabel('x', fontsize=13)
    
    additional_info='x_L1'+str(xL1)+ 'x_L2'+str(xL2)+'kappa'+str(kappa)+'gamma'+str(gamma)
    plt.title(lens_model+ ' -- Contour plot of time delay', fontsize=13)
    plt.legend()
    plt.savefig('../plot/contour_plot_with_points'+additional_info+'.png')
    plt.show()