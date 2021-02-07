
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
import matplotlib.pyplot as plt
import time 
from scipy import signal
# Self-defined package
sys.path.insert(0,os.path.realpath('..')) 
from Images import TFunc, dTFunc, Images
from Fouriertrans import Fd_w,FT_clas

     


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
            print(">>>> a min image")
        # max 
        elif row['typeI'] == 'max':
            tmp[tau_list>=row['tauI']] = -2.*np.pi*(row['muI']**0.5)
            print(">>>> a max image")
        # saddle 
        elif row['typeI'] == 'saddle':
            tmp = -2.*((-row['muI'])**0.5)*np.log(np.absolute(tau_list-row['tauI']))
            print(">>>> a saddle image")
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
        x2_list = x2_bad + np.tile([0.25*dx2, 0.25*dx2, -0.25*dx2, -0.25*dx2], N_bad)
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
    
    
    # avoid edge effects
    tau_list = tau_list[:-10]
    Ft_list = Ft_list[:-10]

    # calculate signular part
    Ftc = FtSingularFunc(images_info, tau_list)
    plt.plot(tau_list,Ftc, label='ftc')
    plt.plot(tau_list,Ft_list,label='ft')
    plt.legend()
    plt.show()
    # remove signular part
    Ftd = Ft_list - Ftc
    tauI-= tImin
    tshift= tImin
    print('first image',tImin)
    print('min value hist', np.min(tau_list))

    return tau_list, Ftd, Ft_list, muI,tauI,typeI,tshift

def fit_Func(t_ori,Ft_orig,funct, tauI):
    
    '''
    fitting of the smoothed curve
    '''

def PutLabels (x_label, y_label, title):
    #plt.style.use('ggplot')
    plt.rcParams["figure.figsize"] = (6,6)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    params = {'axes.labelsize': 16,
              'axes.titlesize': 16,
              'xtick.labelsize' : 16,
              'ytick.labelsize' : 16,
              'font.size':16,
              'legend.fontsize':12,
              'lines.markersize':6
             }
    plt.rcParams.update(params)
    plt.tight_layout()
    
def geom_optics(T,m,t, Itype):  
    
        if Itype=='min':    
            return np.pi*(m**0.5)
        elif Itype=='max':
            return -np.pi*(m**0.5)
        elif Itype=='saddle':
            return -abs(m)**(0.5)*np.log(abs(t-T))
        else:
            raise Exception("Unsupported image type {:} !".format(typeI))
        
        
        
def Phase(F):
    
    return [float(-1j*np.log(complex(r)/abs(complex(r))))  for r in F]



def HistMethod(xL,kappa,gamma, lens_model):
    
    xL1,xL2=xL[0],xL[1]
    
    #info for plots and saved files
    add_info=lens_model+'_x1_'+str(xL1)+ '_x2_'+str(xL2)+'_k_'+str(kappa)+'_g_'+str(gamma)
    path='./2D_Results/'+lens_model
    text_shear=' with external shear'
    if gamma!=0 or kappa!=0:
        title=lens_model+text_shear
    else:
        title=lens_model      
    coord='$x_1$='+str(xL1)+' $x_2$='+str(xL2)
    shear='$\gamma$='+str(gamma)+' $\kappa$='+str(kappa)
    
    if not os.path.exists(path):
            os.makedirs(path)
    
    # accuracy
    tlim = 0.5 
    dt = 1e-3
    
    print('start running...')
    start = time.time()
    tau_list, Ftd, Ft_list, muI, tauI, typeI,tshift = FtHistFunc([xL1, xL2], lens_model, kappa, gamma, tlim, dt)
    time=time.time()-start
    print('finished in', time) 
    
    
    dt=np.diff(tau_list)[-1]
    print ('critical points magnification and time' +muI, tauI)
    
  
    
################ Plot F   #############################################
    
    PutLabels('t','F(t)', title )    
    outfile = './2D_Results/'+lens_model+'/Ft'+add_info+'.png'
    plt.plot(tau_list, abs(Ft_list), '-', c='tab:blue')
    
    print(np.where(tau_list==tauI[0]))
    for i,(T,m) in enumerate(zip(tauI, muI)):
        tmp=np.abs(np.array(tau_list) - T).argmin()
        plt.scatter(T,Ft_list[tmp], c='r')
        
    plt.figtext(.2, .8, coord, fontsize=12)
    plt.figtext(.2, .75, shear, fontsize=12)
    
    plt.savefig(outfile, dpi=100)
        
    plt.show()
    plt.close()

    print('Plot saved to', outfile)

################ Plot F(t) extrapolated at high t   ########################

    PutLabels('t','F(t)','Extrapolation at high t')   
    outfile=path+'/Ft_extended'+add_info+'.png'
    asymptote=np.zeros_like(text)
    for T,m, ty in zip(tauI, muI,typeI):
        asymptote+=geom_optics(T,m,text,ty) 
    
    hshift=Ft_list[-1]-asymptote[0]
    asymptote+=hshift
    
    Ft_new = np.concatenate([Ft_list, asymptote[1:]])
    t_new= np.concatenate([tau_list, text[1:]])

################       Windowing               ##############################
    
    window = signal.cosine(2*len(t_new))    #create the window
    Ft_wind=Ft_new*window[int(window.size/2.):] #apply it
           
################ Plot magnification factor   ##############################    
    

    w,F_diff=Fd_w(t_new,Ft_new,tau_list,Ftd)   #Fourier transform  
    
    w_range=np.round(np.linspace(0.001,100,1050),5)
    #semi-classical contribution
    F_clas=np.zeros((4,len(w_range)), dtype="complex_")

    for i,(m,t, tyI) in enumerate(zip(muI,tauI, typeI)):
        F_clas[i,:]=FT_clas(w_range,t,m, tyI)

    F_clas=np.sum(F_clas,axis=0) 
    Fphase=Phase(F_diff)
    
    df = pd.DataFrame(list(zip(F_diff,Fphase,w)),columns=['Famp','Fphase','w'] )
    df.to_csv(path+'/F/'+lens_model+'Histcount_'+add_info+'.txt', sep='\t')
    
    lim=w<100 
    
    PutLabels('w','|F(w)|', title)     
    plt.plot(w[lim], np.abs(F_diff)[lim], st,label='Hist counting',  c='tab:blue')  
    plt.figtext(.2, .2, coord, fontsize=12 )
    plt.figtext(.2, .15, shear, fontsize=12)   
    plt.xscale('log') 
    plt.legend(loc=2)
    plt.savefig(path+'/2DAmp_'+add_info+'.png',dpi=300)
    plt.show()


    PutLabels('w','$\Phi_F$', title)   
    plt.plot(w_range, Phase(F_clas), 'k--',label='F semi-classical', linewidth=0.5)    
    plt.figtext(.2, .2, coord, fontsize=12)
    plt.figtext(.2, .15, shear, fontsize=12)        
    plt.plot(w[lim],Fphase[lim], st,label='Hist counting',  c='tab:blue') 
    plt.xscale('log')
    plt.xlim(0,100)
    plt.legend(loc=2)
    plt.savefig(path+'/2DPhase_'+add_info+'.png',dpi=300)
    plt.show()
    
if __name__ == '__main__':

    
    # lens
    lens_model = 'SIS'
    xL1 = 0.1
    xL2 = 0.1

    # external shear
    kappa = 0
    gamma = 0.5

    # accuracy
    tlim = 0.5 
    dt = 1e-3
    
    add_info=lens_model+'_x1_'+str(xL1)+ '_x2_'+str(xL2)+'_k_'+str(kappa)+'_g_'+str(gamma)
    path='./2D_Results/'+lens_model


    print('start running...')
    start = time.time()
    tau_list, Ftd, Ft_list, muI, tauI, typeI,tshift = FtHistFunc([xL1, xL2], lens_model, kappa, gamma, tlim, dt)
    time=time.time()-start
    print('finished in', time) 
    
    np.save('muI.npy', muI)
    np.save('tau_list.npy', tau_list)
    np.save('tauI.npy', tauI)
    np.save('Ftd.npy', Ftd)
    np.save('Ft_list.npy', Ft_list)
    np.save('tshift.npy', tshift)
    np.save('typeI.npy', typeI)
    
    
    '''
    muI = np.load('muI.npy')
    tauI = np.load('tauI.npy')
    tau_list= np.load('tau_list.npy')
    Ftd = np.load('Ftd.npy')
    Ft_list= np.load('Ft_list.npy')
    tshift= np.load('tshift.npy')
    typeI= np.load('typeI.npy')
    '''
    
    dt=np.diff(tau_list)[-1]
    print (muI, tauI)
    text_shear=' with external shear'
    if gamma!=0 or kappa!=0:
        title=lens_model+text_shear
    else:
        title=lens_model
        
    coord='$x_1$='+str(xL1)+' $x_2$='+str(xL2)
    shear='$\gamma$='+str(gamma)+' $\kappa$='+str(kappa)
    
################ Plot F   #############################################
    
    PutLabels('t','F(t)', title )    
    outfile = './2D_Results/'+lens_model+'/Ft'+add_info+'.png'
    plt.plot(tau_list, abs(Ft_list), '-', c='tab:blue')
    
    print(np.where(tau_list==tauI[0]))
    for i,(T,m) in enumerate(zip(tauI, muI)):
        tmp=np.abs(np.array(tau_list) - T).argmin()
        plt.scatter(T,Ft_list[tmp], c='r')
        
    plt.figtext(.2, .8, coord, fontsize=12)
    plt.figtext(.2, .75, shear, fontsize=12)
    
    plt.savefig(outfile, dpi=100)
        
    plt.show()
    plt.close()

    print('Plot saved to', outfile)


################ Plot F(t) extrapolated at high t   ########################
  
    
    PutLabels('t','F(t)','Extrapolation at high t')   
    outfile='./2D_Results/'+lens_model+'/Ft_extended'+add_info+'.png'
    
    plt.plot(tau_list,Ft_list, label='original', c='tab:blue')
    text=np.arange(tau_list[-1],10,dt)
    asymptote=np.zeros_like(text)
    for T,m, ty in zip(tauI, muI,typeI):
        asymptote+=geom_optics(T,m,text,ty) 
    #asymptote+=0.5*np.sqrt(xL1**2.+xL2**2)
    plt.plot(text,asymptote,'--' ,label='geom',c='cyan' )   
    
    hshift=Ft_list[-1]-asymptote[0]
    plt.plot(text,asymptote+hshift, label='geom + shift',  c='cyan')
    asymptote+=hshift
    
    Ft_new = np.concatenate([Ft_list, asymptote[1:]])
    t_new= np.concatenate([tau_list, text[1:]])
    print(len(t_new),len(Ft_new))
    #plt.plot(t_new,Ft_new, label='final')
    
    
    if False:
        mask_asymptote=Ft_new>1
        asymptote=np.ones(len(Ft_new[~mask_asymptote]))
        Ft_new = np.concatenate([Ft_new[mask_asymptote],asymptote])
        plt.plot(t_new,Ft_new, '-', label='normalized')
        #plt.show()
    
    
    
    #plt.plot(t_new,Ft_new,'.', label='final', lw=0.3)
    
    plt.figtext(.2, .2, coord, fontsize=12)
    plt.figtext(.2, .15, shear, fontsize=12)
    
    plt.legend()  
    #plt.xlim(0,100)
    plt.ylim(bottom=0)
    plt.savefig(outfile,dpi=300)
    plt.show()
    plt.close()
    

################       Windowing               ##############################
    
    
    PutLabels('x','y','Window Function')   
    window = signal.cosine(2*len(t_new))    #create the window
    Ft_wind=Ft_new*window[int(window.size/2.):] #apply it
    plt.plot(t_new,window[int(window.size/2.):])
    plt.savefig('./2D_Results/Window.png',dpi=300)
    plt.show()
    plt.close()
    
    
    PutLabels('t','F(t)','F(t) -- Windowed')   
    plt.plot(t_new, Ft_new,label='original', c='tab:blue')    
    plt.plot(t_new, Ft_wind,label='windowed', c='cyan')

    #plt.ylim(0,2)
    plt.legend()
    plt.show()
    
    Ft_new=Ft_wind    
    #Ft_new=np.ones(len(Ft_new))
    
    
    
################ Plot magnification factor   ##############################    
    
    PutLabels('w','|F(w)|', title)  
    w,F_diff=Fd_w(t_new,Ft_new,tau_list,Ftd)    
    
    w_range=np.round(np.linspace(0.001,100,1050),5)
    
    
    #phase shift due to the first image
    #F_diff*=np.exp(-1j*tshift*w)
    
    
    #Classical contribution
    F_clas=np.zeros((4,len(w_range)), dtype="complex_")
    #www=np.zeros((4,len(w_range)), dtype="complex_")
    for i,(m,t, tyI) in enumerate(zip(muI,tauI, typeI)):
        F_clas[i,:]=FT_clas(w_range,t,m, tyI)
        #www[i,:]=FT_clas(w_range,t,m)
    #www=np.sum(www,axis=0)
    F_clas=np.sum(F_clas,axis=0)
    

    
    Fphase=np.angle(F_diff)
    
    df = pd.DataFrame(list(zip(F_diff,Fphase,w)),columns=['Famp','Fphase','w'] )
    df.to_csv('./2D_Results/'+lens_model+'/F/'+lens_model+'Histcount_'+add_info+'.txt', sep='\t')
    
    lim=w<100
    
    if lens_model=='point' and xL1==0.1 and xL2==0.1:
        #plot analytical
        
        #wa = np.arange(0.01, 200, 0.001)
        #Fwa = np.loadtxt('./test/Fw_analytical.txt', dtype='cfloat')
        dfpoint=pd.read_csv('./Analytic_pointmass_lens_dist_0.14.txt', sep="\t")
        amp=dfpoint.Famp.values
        wa=dfpoint.w.values
        plt.plot(wa, amp, label='analytical - no shear', c='cyan',lw=0.5)
        
        
    elif lens_model=='SIS' and xL1==0.1 and xL2==0.1:
        
        df_b0=pd.read_csv('./Results/SIScore/Levin_SIScore_lens_dist_0.141_a_1_b_0_c_1.txt', sep="\t")
        amp_b0=[float(abs(complex(i))) for i in df_b0.res_adaptive.values]
        wa=np.linspace(0.001,100,1000)
        plt.plot(wa, amp_b0,label='Levin',c='cyan', lw=0.5)
    
    plt.plot(w_range, np.abs(F_clas), '--',label='F semi-classical',linewidth=0.5, c='darkslategrey')
    
    if gamma==0 and kappa==0:
        st='.'
    else:
        st='-'
        
    plt.plot(w[lim], np.abs(F_diff)[lim], st,label='Hist counting',  c='tab:blue')  
    plt.figtext(.2, .2, coord, fontsize=12 )
    plt.figtext(.2, .15, shear, fontsize=12)
    
    
    plt.xscale('log') 
    #plt.yscale('log')
    plt.legend(loc=2)
    #plt.xlim(0,100)
    plt.savefig(path+'/2DAmp_'+add_info+'.png',dpi=300)
    plt.show()



    PutLabels('w','$\Phi_F$', title)   
    plt.plot(w_range, Phase(F_clas), 'k--',label='F semi-classical', linewidth=0.5)
    #plt.plot(w_range, www )
    
    plt.figtext(.2, .2, coord, fontsize=12)
    plt.figtext(.2, .15, shear, fontsize=12)
    
    
    if lens_model=='point' and xL1==0.1 and xL2==0.1:
        phase=dfpoint.Fphase.values
        plt.plot(wa, phase, label='analytical - no shear', c='cyan',lw=0.5)
        
    elif lens_model=='SIS' and xL1==0.1 and xL2==0.1:
        phase_b0=[float(-1j*np.log(complex(r)/abs(complex(r)))) for r in df_b0.res_adaptive.values]
        plt.plot(wa, phase_b0, label='Levin', c='cyan',lw=0.5)
        
        
    plt.plot(w[lim],Fphase[lim], st,label='Hist counting',  c='tab:blue') 
    plt.xscale('log')
    #plt.axvline(0.1)
    plt.xlim(0,100)
    

    plt.legend(loc=2)
    plt.savefig(path+'/2DPhase_'+add_info+'.png',dpi=300)
    plt.show()
    
    
    
    
    