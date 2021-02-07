
# -*- coding: utf-8 -*-
# @Author: lshuns & mrgr
# @Date:   2020-07-17 15:49:19
# @Last Modified by:   lshuns
# @Last Modified time: 2020-08-30 12:45:03

### solve the diffraction integral in virture of Fourier transform & histogram counting
###### reference: Nakamura 1999; Ulmer 1995

######### Coordinates convention:
############# the origin is set in the centre of perturbation (external shear) and the x-aixs is parallel to the direction of shear, that means gamma2 = 0
############# the source is set in the origin (xs=0)


import numpy as np
import pandas as pd

import os
import sys

# Self-defined package
sys.path.insert(0,os.path.realpath('..')) 
from Images import TFunc, dTFunc, Images
from Fouriertrans import Fd_w,FT_clas
from scipy.interpolate import UnivariateSpline


class TreeClass(object):
    """
    A tree class saving node information.
    Parameters
    ----------
    dt_require: float
        Sampling step in time delay function.
    tlim: float
        Sampling limit in time delay function.    
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

    def SplitFunc(self, weights, x1_list, x2_list, T_list, dT_list):
        """
        Split nodes into bad or good based on the errors of tau values.
        """

        # ++++++++++++ calculate error based on gradient
        error_list = dT_list*(weights**0.5)
        flag_error = (error_list < self.dt_require)

        # +++++++++++++ tlim
        # check the min T in a range (avoid removing potential good nodes)
        T_lim_list = T_list - error_list
        flag_tlim = (T_list < self.tlim)

        # +++++++++++++ total flag
        flag_bad = flag_tlim & np.invert(flag_error)
        flag_good = flag_tlim & flag_error

        # ++++++++++++++ good node information saved to DataFrame
        tmp = pd.DataFrame(data={
            'x1': x1_list[flag_good],
            'x2': x2_list[flag_good],
            'tau': T_list[flag_good],
            'weights': weights*np.ones_like(x1_list[flag_good])
            })
        self.good_nodes = self.good_nodes.append(tmp, ignore_index=True)

        # ++++++++++++++ bad node using simple directory
        self.bad_nodes = [x1_list[flag_bad], x2_list[flag_bad], T_list[flag_bad], dT_list[flag_bad]]


def FtSingularFunc(images_info, tau_list):
    """
    Calculate the singular part of F(t)
    Parameters
    ----------
    images_info: DataFrame
        All images' information.
    tau_list: numpy array
        Sampling of tau where Ftc being calculated.
    """

    # shift tau
    images_info['tauI'] -= np.amin(images_info['tauI'].values)

    Ftc = np.zeros_like(tau_list)
    for index, row in images_info.iterrows():
        tmp = np.zeros_like(tau_list)
        # min 
        if row['typeI'] == 'min':
            tmp[tau_list>=row['tauI']] = 2.*np.pi*(row['muI']**0.5)
            # print(">>>> a min image")
        # max 
        elif row['typeI'] == 'max':
            tmp[tau_list>=row['tauI']] = -2.*np.pi*(row['muI']**0.5)
            # print(">>>> a max image")
        # saddle 
        elif row['typeI'] == 'saddle':
            tmp = -2.*((-row['muI'])**0.5)*np.log(np.absolute(tau_list-row['tauI']))
            # print(">>>> a saddle image")
        else:
            raise Exception("Unsupported image type {:} !".format(row['typeI']))

        Ftc += tmp

    return Ftc

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
    nimages, xI12, muI, tauI, typeI = Images(xL12, lens_model, kappa, gamma, return_mu=True, return_T=True) 
    # collect image info
    images_info = pd.DataFrame(data=
                    {
                    'xI1': xI12[0],
                    'xI2': xI12[1],
                    'muI': muI,
                    'tauI': tauI,
                    'typeI': typeI
                    })

    # tau bounds from images
    tImin = np.amin(images_info['tauI'].values)
    tImax = np.amax(images_info['tauI'].values)

    # tlim is set on the top of tImax
    tlim += tImax

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

    # +++ extend bounds until meeting tlim
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

    # +++ hist counting
    # initial steps
    N_x = 5000

    # initial nodes
    x1_node = np.linspace(boundI_x1[0], boundI_x1[1], N_x)
    x2_node = np.linspace(boundI_x2[0], boundI_x1[1], N_x)
    dx1 = x1_node[1]-x1_node[0]
    dx2 = x2_node[1]-x2_node[0]
    x1_grid, x2_grid = np.meshgrid(x1_node, x2_node)     
    x1_list = x1_grid.flatten()
    x2_list = x2_grid.flatten()
    T_list = TFunc([x1_list, x2_list], xL12, lens_model, kappa, gamma)
    # gradient for error calculation
    dtaudx1, dtaudx2 = dTFunc([x1_list, x2_list], xL12, lens_model, kappa, gamma)
    dT_list = np.sqrt(np.square(dtaudx1)+np.square(dtaudx2))

    # build Tree
    Tree = TreeClass(dt, tlim)
    Tree.SplitFunc(dx1*dx2, x1_list, x2_list, T_list, dT_list)

    # iterate until bad_nodes is empty
    idx = 0
    while len(Tree.bad_nodes[0]):
        idx +=1
        print('loop', idx)

        # bad nodes
        N_bad = len(Tree.bad_nodes[0])
        print('number of bad_nodes', N_bad)

        # +++ subdivide bad nodes' region
        time1 = time.time()
        # each bad node being subdivided to 4 small ones
        x1_bad = np.repeat(Tree.bad_nodes[0], 4)
        x2_bad = np.repeat(Tree.bad_nodes[1], 4)
        # new nodes
        x1_list = x1_bad + np.tile([-0.25*dx1, 0.25*dx1, -0.25*dx1, 0.25*dx1], N_bad)
        x2_list = x2_bad + np.tile([0.25*dx1, 0.25*dx1, -0.25*dx1, -0.25*dx1], N_bad)
        ##
        time2 = time.time()
        # print('list built finished in', time2-time1)

        # +++ calculate tau & dtau
        T_list = TFunc([x1_list, x2_list], xL12, lens_model, kappa, gamma)
        # gradient for error calculation
        dtaudx1, dtaudx2 = dTFunc([x1_list, x2_list], xL12, lens_model, kappa, gamma)
        dT_list = np.sqrt(np.square(dtaudx1)+np.square(dtaudx2))
        ##
        time1 = time.time()
        # print('T_grid and dT_grids finished in', time1-time2)

        # +++ split good and bad
        dx1 *= 0.5
        dx2 *= 0.5
        Tree.SplitFunc(dx1*dx2, x1_list, x2_list, T_list, dT_list)
        # print('Split finished in', time.time()-time1)

    # re-set origin of tau
    Tree.good_nodes['tau'] -= tImin

    # hist counting
    N_bins = int((tlim-tImin)/dt)
    Ft_list, bin_edges = np.histogram(Tree.good_nodes['tau'].values, N_bins, weights=Tree.good_nodes['weights'].values/dt)
    tau_list = (bin_edges[1:]+bin_edges[:-1])/2.
    plt.plot(tau_list,Ft_list)
    
    # avoid edge effects
    tau_list = tau_list[:-10]
    Ft_list = Ft_list[:-10]
    
    '''
    tau_extension,Ft_extension, n_points=fit_Func(tau_list,Ft_list,'ft')
    plt.plot(tau_extension,Ft_extension)
    plt.xlim(0,2)
    plt.show()
    
    index_extension=np.where(tau_extension>np.max(tau_list))[0][0]
    tau_list_extended=np.append(tau_list,tau_extension[index_extension:])
    Ft_list_extended=np.append(Ft_list,Ft_extension[index_extension:])
    plt.plot(tau_list_extended, Ft_list_extended)
    plt.xlim(0,2)
    plt.show()
    '''   
    # calculate signular part
    Ftc = FtSingularFunc(images_info, tau_list)

    # remove signular part
    Ftd = Ft_list - Ftc

    return tau_list, Ftd, Ft_list, muI,tauI

def fit_Func(t_ori,Ft_orig,funct, fit_type_ext='log', fit_type='han'):
    
    '''
    fitting of the smoothed curve
    '''
    dt = 1e-3
      
    if funct=='ftd': 
        
        t_new = np.arange(dt, t_ori[-1], dt)
        
        if fit_type=='lin':
        
            '''
            Fitting of F_d(t) in  order to have a smoother curve and a constant timestep dt.        
            '''
            
            fitting_order=50
            #t_new = np.arange(dt, t_ori[-1], dt)
            z = np.polyfit(t_ori, Ft_orig, fitting_order)
            p = np.poly1d(z)
            Ft_new=p(t_new)
            
            return t_new, Ft_new
        
        elif fit_type=='han':
            
            '''
            hanning smooth to smooth a function
            '''
            window_len=100 #decide the lenght of the window
            s=np.r_[Ft_orig[window_len-1:0:-1],Ft_orig,Ft_orig[-3:-window_len-1:-1]]

            w=np.hanning(window_len)
            Ft_new=np.convolve(w/w.sum(),s,mode='valid')
            Ft_new=Ft_new[int((window_len/2-1)):-int((window_len/2))]
            
            return t_new, Ft_new
            
        
    elif funct=='ftd_ext': 
        
        '''
        Fitting of F_d(t) in  order to extrapolate values at higher times. 
        We can use a linear fitting or a log one.        
        '''
        
        t_max = 100
        #begin_fit=np.where(t_ori>np.max(t_ori)-0.2)[0][0]
        t_cut = 0.6
        tail_mask = t_ori>t_cut
        t_new = np.arange(t_cut,t_max , dt)
    
        
        if fit_type_ext=='lin':
            
            fitting_order=1
            z = np.polyfit(t_ori[tail_mask], Ft_orig[tail_mask], fitting_order)
            p = np.poly1d(z)
            Ft_new=p(t_new)
            
        elif fit_type_ext=='log':
                             
            log_t = np.log(t_ori[tail_mask])
            A = np.vstack([log_t, np.ones_like(log_t)]).T
            m, c = np.linalg.lstsq(A, Ft_orig[tail_mask], rcond=None)[0]
            log_t_new = np.log(t_new)
            Ft_new = m*log_t_new+c
            
        t_final = np.concatenate([t_ori, t_new[t_new>(t_ori[-1]+dt/2)]])
        Ftd_final = np.concatenate([Ft_orig, Ft_new[t_new>(t_ori[-1]+dt/2)]])
    
        return t_final, Ftd_final  
    
    else:
        raise Exception('Unsupported fitting type! using either lin or log!')
        
def PutLabels (x_label, y_label, title):
    #plt.style.use('ggplot')
    plt.rcParams["figure.figsize"] = (5,5)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='lower right',prop={'size': 15})

    
    params = {'axes.labelsize': 16,
              'axes.titlesize': 16,
              'xtick.labelsize' : 16,
              'ytick.labelsize' : 16,
              'font.size':16,
              'lines.markersize':6
             }
    plt.rcParams.update(params)
            

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import time 
    from scipy import signal

    
    # lens
    lens_model = 'point'
    xL1 = 0.1
    xL2 = 0.1

    # external shear
    kappa = 0
    gamma = 0

    # accuracy
    tlim = 0.5
    dt = 1e-3
    
    add_info='_x_L1_'+str(xL1)+ '_x_L2_'+str(xL2)+'_kappa_'+str(kappa)+'_gamma_'+str(gamma)
    
 
    
    print('start running...')
    start = time.time()
    '''
    
    tau_list, Ftd, Ft_list, muI, tauI = FtHistFunc([xL1, xL2], lens_model, kappa, gamma, tlim, dt)
    # print(good_nodes)
    
    
    np.save('muI.npy', muI)
    np.save('tau_list.npy', tau_list)
    np.save('tauI.npy', tauI)
    np.save('Ftd.npy', Ftd)
    np.save('Ft_list.npy', Ft_list)
    print('finished in', time.time()-start)   
    '''
    
    plt.rcParams["figure.figsize"] = (5,5)
    
    muI = np.load('muI.npy')
    tauI = np.load('tauI.npy')
    tau_list= np.load('tau_list.npy')
    Ftd = np.load('Ftd.npy')
    Ft_list= np.load('Ft_list.npy')
    

################ Plot F tilde #############################################
    
    outfile = './plot/'+lens_model+'test_Ft'+additional_info+'.png'
    PutLabels('time','F(t)','Lensed waveform for a delta function pulse')   
    plt.plot(tau_list, Ft_list)
    plt.savefig(outfile, dpi=300)   
    plt.show()
    plt.close()
    print('Plot saved to', outfile)


        
################ Plot F_d   ###############################################
    
    outfile = './plot/'+lens_model+'test_Ftd'+add_info+'.png'
    PutLabels('time','F_d tilde','Diffraction contribution')  
    
    plt.plot(tau_list, Ftd, color='orange', label='original')
    tau_list, Ftd = fit_Func(tau_list,Ftd,'ftd')
    plt.plot(tau_list, Ftd, color='green', label='fitted')

    plt.legend()
    plt.savefig(outfile, dpi=300)
    plt.show()

    print('Plot saved to', outfile)


    
################ Plot F_d extrapolated at high t   ########################
    
  
    t_new, Ft_new=fit_Func(t_smooth, Ftd_smooth,'ftd_ext')
    plt.plot(t_new, Ft_new)
    plt.xlabel('time')
    plt.ylabel('F_d tilde')
    plt.title('F_d tilde extrapolated at high t')
    plt.legend()    
    plt.show()
    plt.savefig('./plot/'+lens_model+'Ftd_extension'+additional_info+'.png',dpi=300)

    
    np.save('t_new.npy', t_new)
    np.save('Ft_new.npy', Ft_new)
    
    
################       Windowing               ##############################

  
    window = signal.cosine(2*len(t_new),40)    #create the window
    Ft_wind=Ft_new*window[int(window.size/2.):] #apply it
    plt.plot(t_new,window[int(window.size/2.):])
    plt.title('Window')
    plt.show()
    
    plt.plot(t_new, Ft_wind,label='windowed')
    plt.plot(t_new, Ft_new,label='original')
    #plt.xlim(0,2)
    plt.legend()
    plt.show()

################ Plot magnification factor   ##############################    
    
    w,F_diff=Fd_w(t_new,Ft_wind,tau_list,Ftd)    
    
    F_clas=np.zeros((4,len(w)), dtype="complex_")
    for i,(m,t) in enumerate(zip(muI,tauI)):
        F_clas[i,:]=FT_clas(w,t,m, t_new, Ft_new)
    F_clas=np.sum(F_clas,axis=0)
    
    
    plt.plot(w, np.abs(F_diff), '.',label='F diffraction')

    plt.plot(w, np.abs(F_clas), label='F semi-classical')
    plt.plot(w, np.abs(F_clas + F_diff), label='F tot')
    plt.xlabel('frequency')
    plt.ylabel('|F|')
    plt.title('Magnification factor')
    
    if gamma==0 and lens_model=='point' and xL1==0.1 and xL2==0.1:
        #plot analytical
        
        wa = np.arange(0.01, 200, 0.001)
        Fwa = np.loadtxt('./test/Fw_analytical.txt', dtype='cfloat')
        plt.plot(wa, np.abs(Fwa), label='analytical')
        
        dfpoint=pd.read_csv('pointHistcount__x1_0.1_x2_0.1_kappa_0_gamma_0.txt', sep="\t")
        amp=[float(abs(complex(i))) for i in dfpoint.Famp.values]
        w=dfpoint.w.values
        plt.plot(w,amp,label='direct FT')
        #phase=dfpoint.Fphase.values
        
    plt.xscale('log') 
    plt.yscale('log')
    plt.xlim(0,100)
        
    
    plt.legend()
    plt.savefig('./plot/'+lens_model+'magnification_factor_extension_1_x_L1'+additional_info+'.png',dpi=300)
    plt.show()

    