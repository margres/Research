# -*- coding: utf-8 -*-
# @Author: lshuns & mrgr
# @Date:   2020-07-17 15:49:19
# @Last Modified by:   lshuns
# @Last Modified time: 2020-08-07 16:54:31

### solve the diffraction integral in virture of Fourier transform & histogram counting
###### reference: Nakamura 1999; Ulmer 1995

######### Coordinates convention:
############# the origin is set in the centre of perturbation (external shear) and the x-aixs is parallel to the direction of shear, that means gamma2 = 0
############# the source is set in the origin (xs=0)

# ++++++++ ISSUES:
#       Cannot go to high tlim
#       Unaccurate for high tau


import numpy as np
import pandas as pd

import os
import sys
# Self-defined package
sys.path.insert(0,os.path.realpath('..')) 
from old_Images import TFunc, Images, saddleORmin
from Fouriertrans import F_d,FT_clas
from scipy.interpolate import UnivariateSpline

class TreeClass(object):
    """
    A tree class saving node information.

    Parameters
    ----------

    """

    def __init__(self, dt_require, tlim):

        # general information
        self.dt_require = dt_require
        self.tlim = tlim

        # initial empty good nodes
        self.good_nodes = pd.DataFrame({'x1': [],
                                        'x2': [],
                                        'tau': [],
                                        'weights': []
                                        })

    def SplitFunc(self, weights, x1_grid, x2_grid, T_grid, singleORwhole):
        """
        Split nodes into bad or good based on the errors of tau values.
        """

        # +++++++++++++ Bounds
        if singleORwhole == 'single':
            # each node has bounds (for extended nodes)
            tmp = np.ones([5,5], dtype=bool)
            tmp[1:-1,1:-1] = np.invert(tmp[1:-1,1:-1])

            bounds_flag = np.tile(tmp, int(T_grid.shape[1]/5))

        elif singleORwhole == 'whole':
            # bounds for a whole matrix (for initial grid)
            tmp = np.ones_like(T_grid, dtype=bool)
            tmp[1:-1,1:-1] = np.invert(tmp[1:-1,1:-1])

            bounds_flag = tmp
            
        else:
            raise Exception("Unknown singleORwhole values!")

        # really used
        flag_nonBounds = np.invert(bounds_flag)

        # +++++++++++++ tlim
        flag_tlim = (T_grid < self.tlim)

        # +++++++++++++ dtau
        # initial grid of flag
        flag_grid = np.zeros_like(T_grid)

        # evaluate accuracy
        dtau_row = np.absolute(T_grid[:, :-1] - T_grid[:, 1:])
        dtau_col = np.absolute(T_grid[:-1, :] - T_grid[1:, :])

        # flag insufficient dtau
        flag_row = dtau_row >= self.dt_require
        flag_col = dtau_col >= self.dt_require

        # flip nodes that do not meet accuracy
        ## right
        flag_grid[:, :-1] += flag_row    
        ## left
        flag_grid[:, 1:] += flag_row
        ## bottom
        flag_grid[:-1, :] += flag_col
        ## top
        flag_grid[1:, :] += flag_col
        ## only those pass all (=0) are accurate enough
        flag_dtau_big = (flag_grid > 0)
        flag_dtau_small = np.invert(flag_dtau_big)

        # +++++++++++++ total flag
        flag_bad = flag_nonBounds & flag_tlim & flag_dtau_big
        flag_good = flag_nonBounds & flag_tlim & flag_dtau_small

        # ++++++++++++++ good node information saved to DataFrame
        tmp = pd.DataFrame(data={
            'x1': x1_grid[flag_good].flatten(),
            'x2': x2_grid[flag_good].flatten(),
            'tau': T_grid[flag_good].flatten(),
            'weights': weights*np.ones(len(x1_grid[flag_good].flatten()))
            })
        self.good_nodes = self.good_nodes.append(tmp, ignore_index=True)

        # ++++++++++++++ bad node simply directory
        self.bad_nodes = [x1_grid[flag_bad].flatten(), x2_grid[flag_bad].flatten()]


def FtHistFunc(xL12, lens_model, kappa=0, gamma=0, tlim=6., dt=1e-2):
    """
    Calculate F(t) with histogram counting

    Parameters
    ----------
    xL12: a list of 1-d numpy arrays [xL1, xL2]
        lens center position, coordinates in the lens plane.
    lens_model: str
        Lens model, supported model ('point').
    kappa: float (optional, default=0)
        Convergence of external shear.
    gamma: float (optional, default=0)
        Shear of external shear.
    tlim: float (optional, default=10.)
        Sampling limit in time delay function (tmax = tI_max+tlim).
    dt: float (optional, default=1e-2)
        Sampling step in time delay function.
    """

    # calculate the images
    nimages, xI12, muI, tauI = Images(xL12, lens_model, kappa, gamma, return_mu=True, return_T=True) 
    # collect image info
    images_info = pd.DataFrame(data=
                    {
                    'xI1': xI12[0],
                    'xI2': xI12[1],
                    'muI': muI,
                    'tauI': tauI
                    })
    # sort images
    try:
    # new for version 1.0.0
        # sort by tau
        images_info_tau = images_info.sort_values(by=['tauI'], ignore_index=True)
        # sort by xI2 (for dividing)
        images_info_xI2 = images_info.sort_values(by=['xI2'], ignore_index=True)

    except:
    # old
        # sort by tau
        images_info_tau = images_info.sort_values(by=['tauI'])
        images_info_tau = images_info_tau.reset_index(drop=True)
        # sort by xI2 (for dividing)
        images_info_xI2 = images_info.sort_values(by=['xI2'])
        images_info_xI2 = images_info_xI2.reset_index(drop=True)
    # print(images_info)
    # print(images_info_tau)
    # print(images_info_xI2)

    # tlim
    tlim += images_info_tau['tauI'].values[-1]
    # print(tlim)

    # initial guess of bounds
    ## x1
    xI1_min = np.amin(images_info['xI1'].values)
    xI1_max = np.amax(images_info['xI1'].values)
    dxI1 = xI1_max - xI1_min
    boundI_x1 = [xI1_min-dxI1, xI1_max+dxI1]
    ## x2
    xI2_min = np.amin(images_info['xI2'].values)
    xI2_max = np.amax(images_info['xI2'].values)
    dxI2 = xI2_max - xI2_min
    boundI_x2 = [xI2_min-dxI2, xI2_max+dxI2]

    # extend bounds until meeting tlim
    while True:
        N_bounds = 1000
        # build the bounds
        tmp2 = np.linspace(boundI_x1[0], boundI_x1[1], N_bounds)
        x1_test = np.concatenate([
                        tmp2,                            # top
                        np.full(N_bounds, boundI_x1[1]), # right
                        tmp2,                            # bottom
                        np.full(N_bounds, boundI_x1[0])  # left 
                        ])

        tmp2 = np.linspace(boundI_x2[0], boundI_x2[1], N_bounds)
        x2_test = np.concatenate([
                        np.full(N_bounds, boundI_x2[1]), # top
                        tmp2,                            # right
                        np.full(N_bounds, boundI_x2[0]), # bottom
                        tmp2                             # left 
                        ])

        # evaluate tau 
        T_tmp = TFunc([x1_test, x2_test], xL12, lens_model, kappa, gamma)

        # break condition
        if np.amin(T_tmp) > tlim:
            break

        # extend bounds
        boundI_x1 = [boundI_x1[0]-0.5*dxI1, boundI_x1[1]+0.5*dxI1]
        boundI_x2 = [boundI_x2[0]-0.5*dxI2, boundI_x2[1]+0.5*dxI2]

    # initial steps
    N_x = 10000

    # initial nodes
    x1_node = np.linspace(boundI_x1[0], boundI_x1[1], N_x)
    x2_node = np.linspace(boundI_x2[0], boundI_x1[1], N_x)
    dx1 = x1_node[1]-x1_node[0]
    dx2 = x2_node[1]-x2_node[0]
    x1_grid, x2_grid = np.meshgrid(x1_node, x2_node)     
    T_grid = TFunc([x1_grid, x2_grid], xL12, lens_model, kappa, gamma)

    # build Tree
    Tree = TreeClass(dt, tlim)
    Tree.SplitFunc(dx1*dx2, x1_grid, x2_grid, T_grid, 'whole')
    # Tree.InitialNotesFunc(dx1*dx2, x1_grid, x2_grid, T_grid, 'whole')
    # Tree.SplitFunc(T_grid)

    # iterate until bad_nodes is empty
    idx = 0
    # while not Tree.bad_nodes.empty:
    while len(Tree.bad_nodes[0]):
        #
        idx +=1
        print('loop', idx)
        # extend nodes
        x1_grids = []
        x2_grids = []
        # print('number of bad_nodes', len(Tree.bad_nodes.index))
        print('number of bad_nodes', len(Tree.bad_nodes[0]))



        time1 = time.time()

        # N_bad = len(Tree.bad_nodes.index)
        N_bad = len(Tree.bad_nodes[0])


        # duplicate original positions
        # x1_bad = np.repeat(Tree.bad_nodes['x1'].values, 5)
        # x2_bad = np.repeat(Tree.bad_nodes['x2'].values, 5)
        x1_bad = np.repeat(Tree.bad_nodes[0], 5)
        x2_bad = np.repeat(Tree.bad_nodes[1], 5)

        # new grid
        x1_node = x1_bad + np.tile([-dx1, -0.5*dx1, 0, 0.5*dx1, dx1], N_bad)
        x1_grids = np.tile(x1_node, (5, 1))
        ##
        x2_grids = np.tile(x2_bad, (5, 1)) + np.array([-dx2, -0.5*dx2, 0, 0.5*dx2, dx2]).reshape(-1,1)

        # for index, bad_node in Tree.bad_nodes.iterrows():
        #     # new bounds
        #     x1_left = bad_node['x1'] - dx1
        #     x1_right = bad_node['x1'] + dx1
        #     x2_bottom = bad_node['x2'] - dx2
        #     x2_top = bad_node['x2'] + dx2
        #     # new nodes
        #     x1_node = np.linspace(x1_left, x1_right, 5)
        #     x2_node = np.linspace(x2_bottom, x2_top, 5)
        #     # new grid
        #     x1_grid, x2_grid = np.meshgrid(x1_node, x2_node) 
            
        #     # save 
        #     x1_grids.append(x1_grid)
        #     x2_grids.append(x2_grid)
        time2 = time.time()
        print('grid finished in', time2-time1)

        # calculate tau
        T_grids = TFunc([x1_grids, x2_grids], xL12, lens_model, kappa, gamma)
        time1 = time.time()
        print('T_grid finished in', time1-time2)

        # split good and bad
        dx1 = x1_node[1]-x1_node[0]
        dx2 = x2_node[1]-x2_node[0]
        # Tree.InitialNotesFunc(dx1*dx2, x1_grids, x2_grids, T_grids, 'single')
        # Tree.SplitFunc(T_grids)
        Tree.SplitFunc(dx1*dx2, x1_grids, x2_grids, T_grids, 'single')
        print('Split finished in', time.time()-time1)


    return Tree.good_nodes, muI, tauI, nimages


def contribute_min(mu_min):
    return 2*np.pi*np.sqrt(mu_min)

def contribute_saddle(bins, mu_saddle, tau_saddle, step, j):
    return -2 * np.sqrt(-mu_saddle)*np.log((np.abs(np.min(bins)+step*j - tau_saddle)))

def contribute_critical(bins,muI,tauI, nimages=4):
    
    saddle_index, minimum_index=saddleORmin(muI, tauI, nimages)
    
    mu_saddle1=muI[saddle_index[0]]
    mu_saddle2=muI[saddle_index[1]]
    mu_min1=muI[minimum_index[0]]
    mu_min2=muI[minimum_index[1]]
    
    tau_saddle1=tauI[saddle_index[0]]
    tau_saddle2=tauI[saddle_index[1]]
    tau_min1=tauI[minimum_index[0]]
    tau_min2=tauI[minimum_index[1]]
    
    hist_time_step=np.diff(bins)[0]
    contrib=[]
    
    bin_center = bins[:-1] + np.diff(bins) / 2
    
    for j,bin_val in enumerate(bin_center,1):
    #min1_min2   
        if bin_val<tau_min2:
            b_tmp= contribute_min(mu_min1)+contribute_saddle(bins,mu_saddle2,tau_saddle2,hist_time_step,j)+contribute_saddle(bins,mu_saddle1,tau_saddle1,hist_time_step,j) 
        
    #min1_saddle1
        elif tau_min2<=bin_val<tau_saddle1:
            
            b_tmp=contribute_min(mu_min2) +contribute_min(mu_min1)+contribute_saddle(bins,mu_saddle2,tau_saddle2,hist_time_step,j)+contribute_saddle(bins,mu_saddle1,tau_saddle1,hist_time_step,j)
              
    #saddle1_saddle2
        elif tau_saddle1 <=bin_val<tau_saddle2:

            b_tmp= contribute_min(mu_min2) +contribute_min(mu_min1)+ contribute_saddle(bins,mu_saddle2,tau_saddle2,hist_time_step,j)+contribute_saddle(bins,mu_saddle1,tau_saddle1,hist_time_step,j)
                  
    #saddle2--
        elif bin_val>=tau_saddle2:
   
            b_tmp=contribute_min(mu_min2)+ contribute_min(mu_min1)+contribute_saddle(bins,mu_saddle1,tau_saddle1,hist_time_step,j)+contribute_saddle(bins,mu_saddle2,tau_saddle2,hist_time_step,j)     
                
        contrib.append(b_tmp)
        
        #print(j*hist_time_step)
    #print(len(bins), np.min(bins), np.max(bins))
    #print(len(contrib))
                
    return bin_center, contrib

def fit_wo_critical(a,b, bins):
    
    '''
    fitting of the smoothed curve
    '''
    
    s = UnivariateSpline(a, b, s=100)
    n_sample=100
    xs = np.linspace(np.min(bins), np.max(bins), n_sample)
    #print(np.max(tau))
    ys = s(xs)
    
    return xs,ys, n_sample


def magnification(F):
    return np.abs(F)**2

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import time 
    
    # lens
    lens_model = 'point'
    xL1 = 0.1
    xL2 = 0.1

    # external shear
    kappa = 0
    gamma = 0.2

    # accuracy
    tlim = 0.5
    dt = 1e-3
    
    
    '''
    print('start running...')
    start = time.time()
    good_nodes,muI, tauI, nimages = FtHistFunc([xL1, xL2], lens_model, kappa, gamma, tlim, dt)
    
    print('finished in', time.time()-start)
    
    
    good_nodes.to_pickle('good_nodes.pkl') 
    np.save('muI.npy', muI)
    np.save('tauI.npy', tauI)
    '''
    
    good_nodes = pd.read_pickle('good_nodes.pkl')
    muI = np.load('muI.npy')
    tauI = np.load('tauI.npy')
    
    
    N_z = int((np.amax(good_nodes['tau'].values)-np.amin(good_nodes['tau'].values))/dt)
    values, bins, _ = plt.hist(good_nodes['tau'].values, N_z, color='gray', weights=good_nodes['weights'].values/dt)
    plt.close()
    #plt.savefig('./test_histT.png',dpi=300)
    
    bin_center, contrib = contribute_critical(bins,muI,tauI)
    
    #print(contrib)
    plt.plot(bin_center,values-contrib)
    plt.xlabel('time')
    plt.ylabel('F tilde')
    
    
    xs, ys,n_sample=fit_wo_critical(bin_center,values-contrib, bins)
    plt.plot(xs, ys)
    plt.show()
    
   
    omega,F_diff=F_d(xs,ys)
    F_clas=np.zeros((4,len(omega)), dtype="complex_")
    
    np.save('xs.npy', xs)
    np.save('ys.npy', ys)
    
    for i,(m,t) in enumerate(zip(muI,tauI)):
        #print(m,t)
        F_clas[i,:]=FT_clas(omega,t,m, xs, ys)
        #print(F_crit[i,:])
    F_clas=np.sum(F_clas,axis=0)
    
    
    pos_indices=np.where(omega<0)[0][1]-1
    plt.plot(omega[:pos_indices],magnification(F_diff[:pos_indices]), label='F diffraction' )
    #plt.plot(omega[:pos_indices],magnification(F_clas[:pos_indices]), label='F semi-classical')
    plt.xlabel('wavelenght')
    plt.ylabel('|F|^2 amplification factor')
    plt.title(str(n_sample)+' sample points')
    #plt.plot(omega,magnification(F_clas), label='F semi-classical')
    plt.legend()
    plt.savefig('./F_d_fit_'+str(n_sample)+'points.png',dpi=300)
    plt.legend()
    plt.show()

    
